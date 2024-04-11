import warnings
warnings.filterwarnings('ignore')
import torch.nn.init as init
import scipy.sparse as sp
import torch_scatter
import torch.nn.functional as F
#from .layers import GraphAttentionLayer, SpGraphAttentionLayer
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat

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
        self.alpha = alpha  # parameter of leakyReLU
        self.concat = concat
        # Define trainable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = torch.mm(h, self.W)  # ((batch*slice*roi,hidden_dim))
        e = self._prepare_attentional_mechanism_input(Wh)  # (batch*slice*roi,batch*slice*roi)
        #
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention1 = F.softmax(attention, dim=1)  # to get normalized attention weight
        attention = F.dropout(attention1, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (batch*slice*roi,hidden_dim))

        if self.concat:
            return F.elu(h_prime), attention1
        else:
            return h_prime, attention1

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

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
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

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
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


#################
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
def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def normalization(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)

    return tensor_adjacency


def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def k_largest_index_argsort(a, k):

    idx = np.argsort(a.ravel())[:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))
def k_smallest_index_argsort(a, k):

    idx = np.argsort(a.ravel())[::-1][:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))
def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1

    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


import torch
from random import randrange

def process_dynamic_fc(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    dynamic_fc_list = []
    for i in sampling_points:
        fc_list = []
        for _t in timeseries:
            fc = corrcoef(_t[i:i+window_size].T)
            if not self_loop: fc -= torch.eye(fc.shape[0])
            fc_list.append(fc)
        dynamic_fc_list.append(torch.stack(fc_list))
    return torch.stack(dynamic_fc_list, dim=1), sampling_points


# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c

#######modularity loss####

import itertools
import random
import torch
from scipy.spatial.distance import cosine

a1 = [29, 30, 41, 42, 71, 72, 73, 74, 75, 76, 81, 82, 83, 84, 87, 88]  # original index of ROIs in SN
a2 = [23, 24, 25, 26, 31, 32, 35, 36, 37, 38, 39, 40, 61, 62, 63, 64, 65, 66, 67, 68, 89, 90]  # DMN
a3 = [3, 4, 7, 8, 59, 60, 13, 14, 15, 16]  # CEN
list1 = torch.tensor(a1) - 1
list2 = torch.tensor(a2) - 1
list3 = torch.tensor(a3) - 1
m = 0.5
IDX1 = list(itertools.combinations(list1, 2))  # pairwise index in modularity1
IDX1 = random.choices(IDX1, k=int(len(IDX1) * m))
IDX2 = list(itertools.combinations(list2, 2))  # pairwise index in modularity2
IDX2 = random.choices(IDX2, k=int(len(IDX2) * m))
IDX3 = list(itertools.combinations(list3, 2))  # pairwise index in modularity3
IDX3 = random.choices(IDX3, k=int(len(IDX3) * m))
Idx_set = [IDX1, IDX2, IDX3]

def calculateloss(X):
    loss = 0
    for x in X:
        list2 = []
        for j in range(3):
            IDX = Idx_set[j]
            list1 = []
            for a, b in IDX:
                roi1 = x[a]  # (64,)
               # print(roi1.shape)
                roi2 = x[b]  # (64,)
                cos = -torch.cosine_similarity(roi1, roi2, dim=-1)
                list1.append(cos)
            loss1 = torch.sum(torch.stack(list1))
            list2.append(loss1)
        loss2 = torch.sum(torch.stack(list2))
        loss += loss2
    return loss

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        #multi-head
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):#adj(8,116,116)
        adj=abs(adj).to_dense()
        x = F.dropout(x, self.dropout, training=self.training)
        aa= [att(x, adj) for att in self.attentions]
        att_list= []
        for i in range(len(aa)):
            att_list.append(aa[i][1])
        x_list = [] #list 5 each: (batch*slice*roi,hidden_dim)
        for i in range(len(aa)):
            x_list.append(aa[i][0])
        x = torch.cat(x_list, dim=1) #(batch*slice*roi,hidden_dim*nhead)
        x = F.dropout(x, self.dropout, training=self.training)
        x,learned_att = self.out_att(x, adj)  # (928,64)#(batch*slice*roi,hidden_dim)
        x = F.elu(x)
        return x

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred
class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
       # print(attn_matrix.shape)
        return x_attend, attn_matrix


class Module_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, sparsity, dropout, cls_token, readout):
        super(Module_1,self).__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

      #  self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.percentile = Percentile()
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            self.gnn_layers.append(GAT(nfeat=hidden_dim, nhid=hidden_dim,  dropout=0.1,nheads=4,alpha=0.2))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
           # self.linear_layers.append(nn.Linear(hidden_dim, num_classes))


    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []
       # a [4, 15, 116, 116])
        for sample, _dyn_a in enumerate(a):
            #_dyn_a(15,116,116)
            for timepoint, _a in enumerate(_dyn_a):
                #_a (116,116)
                thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)#(2,121426)
        _v = torch.cat(v_list).to(a.device)
        #print(a.shape)#(4,15,116,116)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))


    def forward(self, data):
       # print(data.shape[1])
        a, sampling_points =process_dynamic_fc(data, 60, 30,
                                                              data.shape[1])
       # print(a.shape)                                                    
       # sampling_endpoints = [p + 50for p in sampling_points]
        v = torch.nan_to_num(a.float())
        #[4, 15, 116, 116])
        modularityloss = 0.0
        reconstruct_loss = 0.0
        logit = 0.0
        reg_ortho = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        h = v
        h = rearrange(h, 'b t n c -> (b t n) c')#(6940,180)
        h = self.initial_linear(h)#(6940,64)
        a = self._collate_adjacency(a, self.sparsity)
        weight_mask = a.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        for layer, (G, R, T) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules)):
            h = G(h, a)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            # h_bridge(batch,slice, roi, hidden_dim)
            A_pred = dot_product_decode(h)#(batch*slice*roi,batch*slice*roi)
            reconstruct_loss += F.binary_cross_entropy(A_pred.view(-1).cuda(), a.to_dense().view(-1).cuda(),
                                                       weight=weight_tensor.cuda())
            X = rearrange(h_bridge, 't b n c -> (t b) n c')#(batch*slice,roi, hidden)
            modularityloss += calculateloss(X)
            h_readout, node_attn = R(h_bridge, node_axis=2)#(slice,batch,hidden)
            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
            h_attend, time_attn = T(h_readout)
            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1))
            reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

            latent = self.cls_token(h_attend)
           # logit += self.dropout(L(latent))

            attention['node-attention'].append(node_attn)#note that:garo/sero -based  readout can detect spatial attention
            attention['time-attention'].append(time_attn)
            latent_list.append(latent)

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        #latent = torch.stack(latent_list, dim=1)

        return latent, reconstruct_loss,modularityloss


# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()


    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)


    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)


    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
