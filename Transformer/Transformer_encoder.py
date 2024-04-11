#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:28:45 2024

@author: qqw
"""

from einops import rearrange
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
def calculate_metric(gt, pred):
    pred[pred > 0.5] = 1
    pred[pred < 1] = 0
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    acc = (TP + TN) / float(TP + TN + FP + FN)
    sen = TP / float(TP + FN)
    spe = TN / float(TN + FP)
    bac =(sen+spe)/2
    ppv = TP/float(TP+FP)
    npv = TN/float(TN+FN)
    pre =TP/float(TP+FP)
    rec =TP/float(TP+FN)
    f1_score = 2*pre*rec/(pre+rec)
    return acc, sen, spe,bac,ppv,npv,pre,rec,f1_score
def average(args):
  return sum(args, 0.0) / len(args)
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
class ScaledDotProductAttention(nn.Module):
    def __init__(self,hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.hidden_dim)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
def preprocess(data):
    Adj = []
    for i in range(len(data)):
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    return adj
class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_Q = nn.Linear(input_dim, hidden_dim * 4)#n_heads=4
        self.W_K = nn.Linear(input_dim, hidden_dim * 4)
        self.W_V = nn.Linear(input_dim, hidden_dim * 4)
        self.linear = nn.Linear(4 * hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, 4, self.hidden_dim).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        #print(q_s)
        k_s = self.W_K(K).view(batch_size, -1, 4, self.hidden_dim).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, 4, self.hidden_dim).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        context, attn = ScaledDotProductAttention(self.hidden_dim)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, 4 * self.hidden_dim) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]
class EncoderLayer(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.enc_self_attn = MultiHeadAttention(input_dim, hidden_dim)
        self.pos_ffn = PoswiseFeedForwardNet(input_dim, hidden_dim)

    def forward(self, enc_inputs):

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V(原始的qkv信息)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)  ###词嵌入层；定义生成矩阵src_vocab_size, d_model
        #self.pos_emb = PositionalEncoding(d_model)#位置编码层
        self.layers = nn.ModuleList([EncoderLayer(input_dim, hidden_dim) for _ in range(2)])#n_layers=2 # 由前馈神经网络和自注意力组成的EncoderLayer层，然后用modulelist进行多个encode的堆叠，
        # 因为后续没有的encoder并没有使用词向量和位置编码，所有抽离出来 堆叠了6个encode
    def forward(self, enc_inputs):# enc_inputs :[batch_size *src_len]
        ## 将词转化为向量 得到的向量的大小： [batch_size x source_len]

        ## 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        #enc_outputs = self.src_emb(enc_inputs)

        ## 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        #enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 其中大于len的不要，小于的填充
       # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)#batch_size x len_q x len_k
        enc_outputs=enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs)#上一层的输出，当做当前层的输入
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Module_1(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super(Module_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(input_dim, hidden_dim)
        self.dense_1 =nn.Linear(input_dim,hidden_dim)
    def forward(self, enc_inputs):
       # print(enc_inputs.shape)
        enc_inputs = preprocess(enc_inputs).cuda().float()
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
       # print(enc_outputs.shape)
        enc_outputs = rearrange(enc_outputs, 'b n c -> (b n) c')
        x =self.dense_1(enc_outputs)
       # print(x.shape)
        x = rearrange(x, '(b n) c -> b n c', b=enc_inputs.shape[0], n=enc_inputs.shape[1])
        return x