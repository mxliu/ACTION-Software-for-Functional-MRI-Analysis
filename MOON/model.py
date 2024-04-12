import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import scipy.sparse as sp
from einops import rearrange
import torch.nn.functional as F
import scipy


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        Args:
        ----------
            input_dim: the dimension of the input feature

            output_dim: the dimension of the output feature

            use_bias : bool, optional

        """
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
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """

        Args:
        ----
            input_dim: input dimension
            hidden_dim: output dimension
            num_classes: category number (default: 2)
        """
        super(Module_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)


    def forward(self, data):
        a,f=preprocess(data)
        # a(nbatch,116,116) f (nbatch,nroi,ninputdim)
        a = a.cpu().numpy()#(nbatch,nroi,nroi)
        adj = scipy.linalg.block_diag(*abs(a))  # (nbatch*nroi,nbatch*nroi)
        adj_csr = sp.csr_matrix(adj)
        adj_nor = normalization(adj_csr)
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c')#(nbatch*nroi,nroi)
        fea = fea.to(torch.float32)
        gcn1 = F.relu(self.gcn1(adj_nor, fea))  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        gcn2 = F.relu(self.gcn2(adj_nor, gcn1)) #(nbatch*nroi,hiddendim)
        x = rearrange(gcn2, '(b n) c -> b n c', b=int(len(adj_nor) / a.shape[1]), n= a.shape[1])
        return x#  (nbatch*nroi,hiddendim)

class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes=2):
        super(Classifier, self).__init__()
        self.l = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        out = self.l(x)
        return out

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(Net, self).__init__()

        self.gcn = Module_1(input_dim, hidden_dim)
        self.c = Classifier(hidden_dim, num_classes)

    def forward(self, x):
        out = self.gcn(x)
        out1 = torch.mean(out,dim=1)
        out2 = self.c(out1)
        return out1, out2
