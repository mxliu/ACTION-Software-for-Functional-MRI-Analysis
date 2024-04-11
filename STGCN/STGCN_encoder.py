#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:53:54 2024

@author: qqw
"""

# change the model parameters following my MDRL model
import pickle
import numpy as np
with open('/home/qqw/Unsupervised_Pretraining/combine_ADHD_ABIDE_MDD_3806subj_TP170_data.pkl', 'rb') as file:
    full_data = pickle.load(file)  # shape  (3806,170,116)
Adj=[]
for i in range(len(full_data)):
    pc=np.corrcoef(full_data[i].T)
    Adj.append(pc)
adjs=np.array(Adj)
#features=np.expand_dims(features,axis=1)
print("features", adjs.shape)
mean_adj=np.mean(adjs,axis=0)
print(mean_adj.shape)


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
import scipy.io
from torch.utils.data import Dataset, DataLoader

#############################################################################################################################
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import numpy as np


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # kernel_size = 1
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):  # x.shape: torch.Size([64, 64, 128, 22])
        assert A.size(0) == self.kernel_size
        x = self.conv(x)  # torch.Size([64, 64, 128, 22])  see as: N*C*H*W
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)  # torch.Size([64, 1, 64, 128, 22])
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # torch.Size([64, 64, 128, 22])
        # 这个公式可以理解为根据邻接矩阵中的邻接关系做了一次邻接节点间的特征融合，输出就变回了(N * M, C, T, V) 的格式进入tcn
        return x.contiguous(), A  # x.contiguous().shape: torch.Size([64, 64, 128, 22])


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,  # (11, 1)
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()
        # print("Dropout={}".format(dropout))
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  # padding = (5, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])  # kernel_size[1] = 1

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),  # kernel_size[0] = 11
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)  # original paper
        # self.tanh = nn.Tanh()  # modified by yuqi-20210715

    def forward(self, x, A):  # x.shape: torch.Size([64, 1, 128, 22])
        res = self.residual(x)
        x, A = self.gcn(x, A)  # x.shape: torch.Size([64, 64, 128, 22])
        x = self.tcn(x) + res  # x.shape: torch.Size
        # print(x.shape)
        return self.relu(x), A  # original paper
        # return self.tanh(x), A  # modified by yuqi 2021-07-15


class Module_1(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` wheretorch.nn
            :math:`N` is batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,**kwargs) -> object:
        super(Module_1,self).__init__()

        edge_importance_weighting=True
        A = mean_adj
        print("A shape", A.shape)  # (116, 116)
        # print(A.shape)
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)  # DAD.shape: (node_num, node_num)

        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # A.shape: torch.Size([1, node_num, node_num])

        # build networks (number of layers, final output features, kernel size)
        spatial_kernel_size = A.size(0)  # spatial_kernel_size = 1**
        temporal_kernel_size = 11  # update temporal kernel size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # kernel_size = (11, 1)
        self.data_bn = nn.BatchNorm1d(1 * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}  # kwargs0: {}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(1, 16, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(16, 16, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(64, 128, kernel_size, 2, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)  # output: [1, 1, 1, 1, ...]
        '''
        when edge_importance_weighting is True, self.edge_importance:
        ParameterList(
            (0): Parameter containing: [torch.FloatTensor of size 1x116x116]
            (1): Parameter containing: [torch.FloatTensor of size 1x116x116]
            (2): Parameter containing: [torch.FloatTensor of size 1x116x116]
            (3): Parameter containing: [torch.FloatTensor of size 1x116x116])
        '''

        self.cls_fcn1 = nn.Conv2d( 2448, 64, kernel_size=1)


    def forward(self, source):  # x.shape: torch.Size([64, 1, W, 22, 1]), where 64 is batch size

        source=torch.unsqueeze(source,dim=1)
        source=torch.unsqueeze(source,dim=4)
        #print(source.shape)
        N, C, T, V, M = source.size()  # N=64, C=1, T=W(e.g., 128), V=22, M=1
        source = source.permute(0, 4, 3, 1, 2).contiguous()  # torch.Size([64, 1, 22, 1, W])
        source = source.view(N * M, V * C, T)  # torch.Size([64*1, 22*1, W])
        source = self.data_bn(source.float()).cuda()  # torch.Size([64*1, 22*1, W])
        source = source.view(N, M, V, C, T)  # torch.Size([64, 1, 22, 1, W])
        source = source.permute(0, 1, 3, 4, 2).contiguous()  # torch.Size([64, 1, 1, W, 22])
        source = source.view(N * M, C, T, V)  # torch.Size(1*64, 1, W, 22]) #(32,16,230,116)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            source, _ = gcn(source, self.A * importance)  # [64, 16, 128, 116]
        # source = F.avg_pool2d(source, source.size()[2:])  # torch.Size([64, 16, 1, 1]), before: [64, 16, 128, 116]

        source = source.mean(axis=3)  # [bs, 16, 200]  #(32,16,230,116)
        source = source.view(source.size(0), -1)  # [bs, 3200]

        #  prediction  #
        source = source.view(N, M, -1, 1, 1).mean(dim=1)
        # print(source.shape)
        # source = self.cls_fcn1(source)
        source = self.cls_fcn1(source)
       # print("source", source.shape)  # source torch.Size([32, 64, 1, 1])
        source = source.squeeze()
       # print("source", source.shape)#(32, 64)

        return source
