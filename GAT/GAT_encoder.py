#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:56:47 2024

@author: qqw
"""

#change the model parameters following my MDRL model
import warnings
warnings.filterwarnings('ignore')
import os
#from model import *
#import util
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import torch_scatter  # 注意：torch_scatter 安装时编译需要用到cuda
import scipy.io
import h5py
from einops import rearrange, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import GraphAttentionLayer, SpGraphAttentionLayer


def k_smallest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[::-1][:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))
class MDD(object):
    def read_data(self):
        HC = scipy.io.loadmat('/home/qqw/A_newfMRI_data/HC.mat')['A']
        # NOANI = scipy.io.loadmat('/home/qqw/newfMRI_data/NOANI.mat')['A']
        ANI = scipy.io.loadmat('/home/qqw/A_newfMRI_data/NOANI.mat')['A']
        #
        import numpy as np
        alldata = np.concatenate((HC, ANI), axis=1)
        A = np.squeeze(alldata.T)
        # y1 = np.zeros(70)
        y2 = np.zeros(70)
        y3 = np.ones(68)
        y = np.concatenate((y2, y3), axis=0)
        adj_list = []
        fea_list = []
        for i in range(len(A)):  # 人数
            signal = A[i]#(232,116)
            pc = np.corrcoef(signal.T)
            pc = np.nan_to_num(pc)
            fea_list.append(pc)
            pc_idx = k_smallest_index_argsort(pc, k=int(0.5 * len(pc) * len(pc)))#original 0.7
            for m, n in zip(pc_idx[:, 0], pc_idx[:, 1]):
                pc[m, n] = 0
            adj_list.append(pc)
            #  print(A[i].shape)
        adj=np.array(adj_list)
        fea=np.array(fea_list)
       # X = np.array(series)  # 148 200 116
        print(adj.shape)#(533, 116, 116)
        print(fea.shape)  # (533, 116, 116)
       # y = site1['lab']
     #   y = np.squeeze(y)
        #print(y.shape)#(533,)
        return adj, fea,y

    def __init__(self):
        super(MDD, self).__init__()
        adj, fea,y = self.read_data()
        
        self.adj = torch.from_numpy(adj)
        self.fea = torch.from_numpy(fea)
        self.y = torch.from_numpy(y)
        self.n_samples = adj.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.adj[index],self.fea[index], self.y[index]


full_dataset = MDD()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # 1160,116 (116,64) h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#train_loader = DataLoader(dataset=full_dataset, batch_size=4, shuffle=True)
# for epoch in range(1):
#     print("epoch: {} -------------------------------------".format(epoch))
# for i,(adj,fea,labels) in enumerate(train_loader):
#     print(i, adj.shape, fea.shape,labels)
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
    bac = (sen + spe) / 2
    ppv = TP / float(TP + FP)
    npv = TN / float(TN + FN)
    pre = TP / float(TP + FP)
    rec = TP / float(TP + FN)
    f1_score = 2 * pre * rec / (pre + rec)
    return acc, sen, spe, bac, ppv, npv, pre, rec, f1_score


def tensor_from_numpy(x, device):  # numpy数组转换为tensor 并转移到所用设备上
    return torch.from_numpy(x).to(device)


def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接 A+I
    degree = np.array(adjacency.sum(1))  # 得到此时的度矩阵对角线 对增加自连接的邻接矩阵按行求和
    d_hat = sp.diags(np.power(degree, -0.5).flatten())  # 开-0.5次方 转换为度矩阵（对角矩阵）
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()  # 得到归一化、并引入自连接的拉普拉斯矩阵 转换为coo稀疏格式
    # 转换为 torch.sparse.FloatTensor
    # 稀疏矩阵非0值 的坐标（行索引，列索引）
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    # 非0值
    values = torch.from_numpy(L.data.astype(np.float32))
    # 存储为tensor稀疏格式
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)

    return tensor_adjacency


def global_max_pool(x, graph_indicator):
    # 对于每个图保留节点的状态向量 按位置取最大值 最后一个图对应一个状态向量
    num = graph_indicator.max().item() + 1
    # print (num)
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))


def k_smallest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[::-1][:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))


def global_avg_pool(x, graph_indicator):
    # 每个图保留节点的状态向量 按位置取平均值 最后一个图对应一个状态向量
    num = graph_indicator.max().item() + 1

    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


def obtain_adjandfea(X):
    # construct proportional graph for each subject
    fc_list = []
    for i in range(len(X)):
        # print(A[i].T.shape)
        pc = np.corrcoef(X[i].cpu().T)
        pc = np.nan_to_num(pc)
        # print(len(pc))
        pc_idx = k_smallest_index_argsort(pc, k=int(0.5 * len(pc) * len(pc)))
        for m, n in zip(pc_idx[:, 0], pc_idx[:, 1]):
            pc[m, n] = 0
        # pc = abs(pc)
        fc_list.append(pc)
    x = np.array(fc_list)
    adj = scipy.linalg.block_diag(*abs(x))  # (3712,3712)
    adj_csr = sp.csr_matrix(adj)
    adj_nor = normalization(adj_csr).to(device)
    adj_nor = adj_nor.to(torch.float32)
    # construct shared adj for all subject
    # x1 = reduce(X, ' a b c->b c', 'mean')
    # from einops import repeat
    # x2 = repeat(x1, 'b c -> a b c', a=len(X))
    # adj = scipy.linalg.block_diag(*abs(x2))
    # adj_csr = sp.csr_matrix(adj)
    # adj_nor = normalization(adj_csr).to(device)
    # adj_nor = adj_nor.to(torch.float32)
    # construct adj for each subject
    fc_list = []
    for i in range(len(X)):
        # print(A[i].T.shape)
        pc = np.corrcoef(X[i].cpu().T)
        pc = np.nan_to_num(pc)
        # pc = abs(pc)
        fc_list.append(pc)
    a = np.array(fc_list)  # (32, 116, 116)
    a_ = abs(a)
    # repectively
    # a_new = reduce(a_, 'a b c -> b c', 'mean')
    # a_nnew = repeat(a_new, ' b c ->a b c', a=len(X))
    # adj = scipy.linalg.block_diag(*a_)  # (3712,3712)
    # adj_csr = sp.csr_matrix(adj)
    # adj_nor = normalization(adj_csr).to(device)
    # adj_nor = adj_nor.to(torch.float32)
    a = torch.from_numpy(a)
    a_ = torch.from_numpy(a_)
    fea = rearrange(a, 'a b c-> (a b) c').to(device)
    # fea=torch.nan_to_num(fea)
    fea = fea.to(torch.float32)
    return adj_nor, fea



def normalization(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        normalized matrix, type torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)

    return tensor_adjacency

def preprocess(data):
    Adj = []
    for i in range(len(data)):
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    fea = adj
    return adj,fea
class Module_1(nn.Module):
    def __init__(self, nfeat, nhid):
        """Dense version of GAT."""
        super(Module_1, self).__init__()
        # self.dropout = dropout
        nheads = 4
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=0.5, alpha=0.1, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=0.5, alpha=0.1, concat=False)
        # self.fc1 = nn.Linear(2* nhid, 2 * nhid // 2)
        #
        # self.fc2 = nn.Linear(2 * nhid // 2, 2)

    def forward(self, data):  # adj(8,116,116)
        a,f=preprocess(data)
        # a(nbatch,116,116) f (nbatch,nroi,ninputdim)
        a = a.cpu().numpy()#(nbatch,nroi,nroi)
        adj = scipy.linalg.block_diag(*abs(a))  # (nbatch*nroi,nbatch*nroi)
        adj_csr = sp.csr_matrix(adj)
        adj_nor = normalization(adj_csr).cuda()
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c').cuda()#(nbatch*nroi,nroi)
        fea = fea.to(torch.float32)
        x = rearrange(f, 'a b c-> (a b) c').cuda()  # (8,116,116)
        # fea=torch.nan_to_num(fea)
        x = x.to(torch.float32)  # (1160,116)
        adj = scipy.linalg.block_diag(*abs(a))  # (1160,1160)
        adj = torch.from_numpy(adj).to(torch.float32).cuda()
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, 0.5, training=self.training)  # 1160,64
        gat1 = F.elu(self.out_att(x, adj))  # (1160,16)
        x = rearrange(gat1, '(b n) c -> b n c', b=int(len(adj) / 116), n=116)  # torch.Size([32, 116, 64])
        # # x = x.view(x.size(0), -1)
        #  x = torch.cat((reduce(x, 'b n c ->b c', 'mean'), reduce(x, 'b n c ->b c', 'max')), dim=1)
        #  x = F.relu(self.fc1(x))
        #  x = self.fc2(x)
        return x  # F.log_softmax(x, dim=1)