
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import os
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
import torch_scatter
import scipy.io
import h5py
from einops import rearrange, reduce


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)  # XW (N,output_dim=hidden_dim)
        output = torch.sparse.mm(adjacency, support)  # L(XW)  (N,output_dim=hidden_dim)
        if self.use_bias:
            output += self.bias
        return output  # (N,output_dim=hidden_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


def normalization(adjacency):

    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)

    return tensor_adjacency


class Module_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):

        super(Module_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)

    def forward(self, a, f):
        # a(nbatch,116,116) f (nbatch,nroi,ninputdim)
        a = a.cpu().numpy()#(nbatch,nroi,nroi)
        adj = scipy.linalg.block_diag(*abs(a))  # (nbatch*nroi,nbatch*nroi)
        adj_csr = sp.csr_matrix(adj)
        adj_nor = normalization(adj_csr).cuda()
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c').cuda()#(nbatch*nroi,nroi)
        # fea=torch.nan_to_num(fea)
        fea = fea.to(torch.float32)
        gcn1 = F.relu(self.gcn1(adj_nor, fea))  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        gcn2 = F.relu(self.gcn2(adj_nor, gcn1))
        # print(gcn1)
        # gcn2 = F.relu(self.gcn2(adjacency, gcn1))  # (N,hidden_dim)#torch.Size([3712, 116])
        # print(gcn2.shape)#
        b = len(adj_nor) / 116
        x = rearrange(gcn2, '(b n) c -> b n c', b=int(len(adj_nor) / a.shape[1]), n= a.shape[1])
       ## print(x.shape)
        #note x[nbatch,nroi,nhidden dim]
        return x
