#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:26:41 2024

@author: qqw
"""

# This is the original GCN code created for MDD_SITE20 data
import warnings

warnings.filterwarnings('ignore')
import os
# from model import *
# import util
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
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
from torch.utils.data.dataset import Dataset
import torch.utils.data.dataloader
import pickle
with open('/home/qqw/Unsupervised_Pretraining/combine_ADHD_ABIDE_MDD_3806subj_TP170_data.pkl', 'rb') as file:
    full_data = pickle.load(file)  # shape  (3806,170,116)
Features=[]
for i in range(len(full_data)):
    pc=np.corrcoef(full_data[i].T)
    Features.append(pc)
features=np.array(Features)
features=np.expand_dims(features,axis=1)
print("features", features.shape)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # 权重矩阵
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 使用自定义参数初始化方式

    def reset_parameters(self):  # 自定义权重和偏置的初始化方式
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"""
        # adjacency (N,N) 归一化的拉普拉斯矩阵
        # input_feature（N,input_dim） N为所有节点个数 （包含所有图）
        support = torch.mm(input_feature, self.weight)  # XW (N,output_dim=hidden_dim)
        output = torch.sparse.mm(adjacency, support)  # L(XW)  (N,output_dim=hidden_dim)
        if self.use_bias:
            output += self.bias
        return output  # (N,output_dim=hidden_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


from sklearn.metrics import confusion_matrix
def preprocess(data):
    Adj = []
    for i in range(len(data)):
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    return adj

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


# Define the E2EBlock class
class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        # print("self.d ",self.d )116
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        x = x.float()
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


# Define the BrainNetCNN class
class Module_1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Module_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        example = torch.from_numpy(features).unsqueeze(1).float()
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 32, example, bias=True)
        self.e2econv2 = E2EBlock(32, 64, example, bias=True)

        self.dense1 = torch.nn.Linear(116, hidden_dim)
        # self.dense2 = torch.nn.Linear(128, 30)
        # self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        x = preprocess(x).cuda().float()
        x=torch.unsqueeze(x,dim=1)
        #print(x.shape)  # torch.Size([32, 1, 116, 116])
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
       # print(out.shape)  # torch.Size([32, 64, 116, 116])
        x = torch.mean(out, dim=1)
        x = self.dense1(x)
        #print("x shape", x.shape)

        return x