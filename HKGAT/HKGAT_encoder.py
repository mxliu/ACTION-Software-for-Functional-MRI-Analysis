import scipy.io
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io
from einops import rearrange, reduce
from pmath import project, logmap0
#import torch_scatter  # 注意：torch_scatter 安装时编译需要用到cuda

print(torch.__version__)  # PyTorch 版本
print(torch.cuda.is_available())  # 是否支持 CUDA
print(torch.version.cuda)  # CUDA 版本
print("torch_scatter installed successfully!")


def k_smallest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[::-1][:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))


class Sine(torch.nn.Module):  # pi初始值3.14*1.2
    def __init__(self, data_dim=-1, phi=3.1415926 * 0.3, bias=False):
        super(Sine, self).__init__()

        if bias is False:
            self.phi = phi
        else:
            self.phi = 0.0

        if data_dim > 0:
            self.A = np.sqrt(1.0 / (2.0 * data_dim))
        else:
            self.A = 1

    def forward(self, x):
        # x = self.A * cosine_activator_.apply(x + self.phi)
        x = self.A * torch.cos(x + self.phi)
        return x


class Cosine(torch.nn.Module):  # pi初始值3.14*1.2
    def __init__(self, data_dim=-1, phi=3.1415926 * 0.3, bias=False):
        super(Cosine, self).__init__()

        if bias is False:
            self.phi = phi
        else:
            self.phi = 0.0

        if data_dim > 0:
            self.A = np.sqrt(1.0 / (2.0 * data_dim))
        else:
            self.A = 1

    def forward(self, x):
        # x = self.A * cosine_activator_.apply(x + self.phi)
        x = self.A * torch.cos(x + self.phi)
        return x


class Cosine_fMRI(torch.nn.Module):  # pi初始值3.14*1.2
    def __init__(self, data_dim=-1, phi=3.1415926 * 0.3, bias=False):
        super(Cosine_fMRI, self).__init__()

        if bias is False:
            self.phi = phi
        else:
            self.phi = 0.0

        if data_dim > 0:
            self.A = np.sqrt(1.0 / (2.0 * data_dim))
        else:
            self.A = 1

    def forward(self, x):
        # x = self.A * cosine_activator_.apply(x + self.phi)
        x = self.A * torch.cos(x + self.phi)
        return x


class FKernel(torch.nn.Module):
    def __init__(self, c):
        super(FKernel, self).__init__()
        # self.device = device
        self.c = c

    def forward(self, x):
        output = project(x, c=self.c)
        output = logmap0(output, c=self.c)
        return output


class WKernel(torch.nn.Module):
    def __init__(self, n_input, n_output, a_prompt):
        super(WKernel, self).__init__()
        # self.device = device
        self.fc0 = torch.nn.Linear(n_input, n_output, bias=True)
        self.fc1 = torch.nn.Linear(n_input, n_output, bias=True)
        self.bn0 = torch.nn.BatchNorm1d(116)  # batchnormlization 应该是对于（137,116,128）中的116
        self.bn1 = torch.nn.BatchNorm1d(116)
        self.sine = Sine(data_dim=n_output)
        self.n_input = n_input
        self.n_output = n_output
        self.a_prompt = a_prompt
        self.init_params()

    def init_params(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # print('self.n_input,self.n_output',self.n_input,self.n_output)
        # print("WKernel x.shape",x.shape)
        linear0 = self.fc0(x)
        # print("WKernel linear0.shape",linear0.shape)
        bn0 = self.bn0(linear0)
        # print("WKernel bn0.shape",bn0.shape)
        x1 = self.sine(bn0)
        # print("WKernel x1.shape",x1.shape)

        # x1 = self.cosine( self.fc0(x) )
        x2 = torch.relu(self.bn1(self.fc1(x)))
        # x2 = self.sine( self.fc1(x) )
        # return x* x1 + x2
        return self.a_prompt * x1 + x2


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, c, a_para_GATLayer, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.c = c
        self.fkernel = FKernel(self.c)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # create an uninitialized tensor(random values will be filled in),of shape (2*out_features, 1). self.a is a learnable value
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a_para_GATLayer = a_para_GATLayer
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.cos = Cosine()

    def forward(self, h, adj):
        # print('h.shape', h.shape, 'adj.shape', adj.shape)
        # print('self.W.shape', self.W.shape)

        Wh = torch.mm(h, self.W)  # 1160,116 (116,64) h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print('Wh.shape',Wh.shape)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        # applying the adjacency matrix, this is a critical operation of attention
        attention = torch.where(adj > 0, e, zero_vec)  # if adj>0. attention=e, elso attention=zero_vec
        attention = F.softmax(attention, dim=1)  # Eq.(3), alpah
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # A part of Eq.(4)
        h_prime = self.fkernel(h_prime)
        # Equation (4) Atcivation
        if self.concat:
            # return F.elu(h_prime) #why use this?
            return F.elu(h_prime) + self.a_para_GATLayer * self.cos(h_prime)
            # return self.leakyrelu(e) + self.a_para_GATLayer* self.cos(e) #leakyrelu, equation (3)  #YMM

        else:
            return h_prime

    # this prepared opperation does not applying the adjacency matrix
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T  # e is asymmetric. This equals to Eq.(1). Why GAT use index_matrix in which the diagonal elements are 0?
        return self.leakyrelu(e) + self.a_para_GATLayer * self.cos(e)  # leakyrelu, equation (3)  #YMM

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


def preprocess(data):
    Adj = []
    for i in range(len(data)):
        # Pearson correlation coefficient matrix.
        # The Pearson correlation between two features measures the linear relationship between them, and its value ranges from -1 to 1
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        # focus only on the magnitude of the relationship, ignoring whether it's positive or negative
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    fea = adj
    return adj, fea


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
    adj_nor = normalization(adj_csr).to(device)  # L=D^-0.5 * (A+I) * D^-0.5,
    adj_nor = adj_nor.to(torch.float32)
    fc_list = []
    for i in range(len(X)):
        # print(A[i].T.shape)
        pc = np.corrcoef(X[i].cpu().T)
        pc = np.nan_to_num(pc)
        # pc = abs(pc)
        fc_list.append(pc)
    a = np.array(fc_list)  # (32, 116, 116)
    a_ = abs(a)
    a_ = torch.from_numpy(a_)
    fea = rearrange(a, 'a b c-> (a b) c')  # .to(device)
    # fea=torch.nan_to_num(fea)
    fea = fea.to(torch.float32)
    return adj_nor, fea


class HGNNEncoder(torch.nn.Module):
    def __init__(self, nfeat, nhid, c, a_couping, a_fMRI, a_DTI, weight_fMRI, weight_DTI):  # (116,dim)

        """

        Args:
        ----
            input_dim: input dimension
            hidden_dim: output dimension
            num_classes: category number (default: 2)
        """
        super(HGNNEncoder, self).__init__()
        nheads = 4
        self.a_couping = a_couping
        self.attentions = [GraphAttentionLayer(nfeat * 4, nhid, c, a_couping, dropout=0.5, alpha=0.1, concat=True) for _
                           in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, c, a_couping, dropout=0.5, alpha=0.1, concat=False)
        self.c = c
        self.a_para = a_couping
        self.weight_fMRI = weight_fMRI
        self.weight_DTI = weight_DTI
        self.fkernel = FKernel(self.c)  # 这里也加了双曲核
        self.cos = Sine()

        self.coupling = coupling(nfeat, nhid, c, a_fMRI, a_DTI, weight_fMRI, weight_DTI)
        # self.transformer = TransformerModel(feature_size, num_layers, nhead)

    def forward(self, DTI, adj_DTI, fMRI):
        # print("test_coupling")
        adj_coupled, DTI_HKGAT, fMRI_HKGAT, adj_DTI, adj_fMRI = self.coupling(DTI, adj_DTI, fMRI)
        # data = self.coupling(DTI, adj_DTI, fMRI)
        adj_fMRI, f_fMRI = preprocess(fMRI)
        f_fMRI = f_fMRI.cuda()

        # 对于HKGAT，可能是由于Attention机制，这里我们的DTI和fMRI是经过L2 Norm归一化得到的结果，对于GCN，我们也可以试试
        epsilon = 1e-8
        DTI_norm = DTI / (DTI.norm(p=2, dim=-1, keepdim=True) + epsilon)
        f_fMRI_norm = f_fMRI / (f_fMRI.norm(p=2, dim=-1, keepdim=True) + epsilon)

        f = torch.cat((self.weight_DTI * DTI_norm, self.weight_fMRI * f_fMRI_norm), dim=-1)  # DTI本身就norm之后相加的
        a = adj_coupled
        a = a.detach().cpu().numpy()
        # a = a.cpu().numpy()#(nbatch,nroi,nroi)
        # print('data.shape',data.shape)
        x = rearrange(f, 'a b c-> (a b) c').cuda()  # (16,116,116)-> (16*116,116)?
        # print('x.shape',x.shape)
        x = x.to(torch.float32)  # (1856,116)

        x = self.fkernel(x)
        # print('a.shape',a.shape)
        adj = scipy.linalg.block_diag(*abs(a))  # (1160,1160)
        adj = torch.from_numpy(adj).to(torch.float32).cuda()

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, 0.5, training=self.training)  # 1160,64
        x1 = self.out_att(x, adj)
        x = self.fkernel(x)

        gat1 = F.elu(x1) + self.a_para * self.cos(x1)  # (1856,64)
        # print('gat1.shape',gat1.shape)
        x = rearrange(gat1, '(b n) c -> b n c', b=int(len(adj) / 116), n=116)  # torch.Size([32, 116, 64])
        # print('x.last.shape',x.shape)
        return x, adj_coupled, DTI_HKGAT, fMRI_HKGAT, adj_DTI, adj_fMRI  # (nbatch*nroi,hiddendim) #将形状为(nbatch*nroi,hiddendim)的张量重新排为(nbatch，nroi,hiddendim)


class coupling(torch.nn.Module):
    def __init__(self, in_channels, feature_size, c, a_fMRI, a_DTI, weight_fMRI, weight_DTI):
        super(coupling, self).__init__()
        self.module1 = Module_DTI(in_channels * 3, feature_size, c, a_DTI)
        self.module2 = Module_fMRI(in_channels, feature_size, c, a_fMRI)
        # def __init__(self, nfeat, nhid, c, a_para):#(116,dim)

        # self.a_fMRI = a_fMRI
        # self.a_DTI = a_DTI
        self.weight_fMRI = weight_fMRI
        self.weight_DTI = weight_DTI

    def forward(self, DTI, adj_DTI, fMRI):
        # utilize the two graph embedding by 2 GCNs
        data_DTI = self.module1(DTI, adj_DTI)
        data_fMRI, adj_fMRI = self.module2(fMRI)  # 或者看一下module_1原来的

        data_fMRI = data_fMRI / data_fMRI.norm(p=2, dim=-1, keepdim=True)
        epsilon = 1e-7
        data_DTI = data_DTI / (data_DTI.norm(p=2, dim=-1, keepdim=True) + epsilon)
        # print('shape of DTI',data_fMRI.shape,'shape of fMRI',data_DTI.shape)
        data_DTI_transposed = data_DTI.transpose(1, 2)  # Shape: (64, 64, 116)
        # print('data_fMRI.shape',data_fMRI.shape,'data_DTI.shape',data_DTI.shape)
        coupled_data = torch.matmul(data_fMRI, data_DTI_transposed)  # Shape will be (64, 116, 128)
        # print('coupled_data.shape',coupled_data.shape)
        return coupled_data, data_DTI, data_fMRI, adj_DTI, adj_fMRI


class Module_DTI(nn.Module):
    def __init__(self, nfeat, nhid, c, a_para):  # (116,dim)
        """Dense version of GAT."""
        super(Module_DTI, self).__init__()
        # self.dropout = dropout
        nheads = 4
        # nfeat and nhid means input_feature and output_feature
        self.attentions = [GraphAttentionLayer(nfeat, nhid, c, a_para, dropout=0.5, alpha=0.1, concat=True) for _ in
                           range(nheads)]
        # The output of GraphAttentionLayer is  F.elu(h_prime), head from 1 to 4
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, c, a_para, dropout=0.5, alpha=0.1, concat=False)
        self.c = c
        self.fkernel = FKernel(self.c)  # 这里也加了双曲核
        self.cos = Cosine_fMRI()
        self.a_para = a_para

    def forward(self, data, adj):  # adj(16,230,116)
        a = adj
        f = data
        # print('a.shape',a.shape,'f.shape',f.shape)

        a = a.cpu().numpy()  # (nbatch,nroi,nroi)(16,116,116)
        x = rearrange(f, 'a b c-> (a b) c').cuda()  # (16,116,116)-> (16*116,116)?
        # print('x.shape',x.shape)
        x = x.to(torch.float32)  # (1856,116)

        # print('a.shape',a.shape)
        adj = scipy.linalg.block_diag(*abs(a))  # (1160,1160)
        # print('adj.shape',adj.shape)

        adj = torch.from_numpy(adj).to(torch.float32).cuda()
        # print('adj2.shape',adj.shape)
        # realize the multihead attention by concatenating the single head attention
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print('x.shape',x.shape)

        x = F.dropout(x, 0.5, training=self.training)  # 1160,64
        # print('x.shape',x.shape)
        x1 = self.out_att(x, adj)
        x1 = self.fkernel(x1)
        gat1 = F.elu(x1) + self.a_para * self.cos(x1)  # (1856,64)

        # gat1 = F.elu(self.out_att(x, adj))  # (1856,64)
        # print('gat1.shape',gat1.shape)
        x = rearrange(gat1, '(b n) c -> b n c', b=int(len(adj) / 116), n=116)  # torch.Size([32, 116, 64])
        # print('x.last.shape',x.shape)
        return x  # F.log_softmax(x, dim=1)


class Module_1(nn.Module):
    def __init__(self, nfeat, nhid, c, a_para):  # (116,dim)
        """Dense version of GAT."""
        super(Module_1, self).__init__()
        # self.dropout = dropout
        nheads = 4
        # nfeat and nhid means input_feature and output_feature
        self.attentions = [GraphAttentionLayer(nfeat, nhid, c, a_para, dropout=0.5, alpha=0.1, concat=True) for _ in
                           range(nheads)]
        # The output of GraphAttentionLayer is  F.elu(h_prime), head from 1 to 4
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, c, a_para, dropout=0.5, alpha=0.1, concat=False)
        self.c = c
        self.fkernel = FKernel(self.c)  # 这里也加了双曲核
        self.cos = Cosine_fMRI()
        self.a_para = a_para

    def forward(self, data):  # adj(16,230,116)
        # print("test_fMRI")
        a_initial, f_initial = preprocess(data)
        # a,f=preprocess(data) #preprocess the graph to obtain the feature and adjacency matrix
        a = a_initial.cpu().numpy()  # (nbatch,nroi,nroi)(16,116,116)
        # print('data.shape',data.shape)
        x = rearrange(f_initial, 'a b c-> (a b) c').cuda()  # (16,116,116)-> (16*116,116)?
        # print('x.shape',x.shape)
        x = x.to(torch.float32)  # (1856,116)

        # print('a.shape',a.shape)
        adj = scipy.linalg.block_diag(*abs(a))  # (1160,1160)
        # print('adj.shape',adj.shape)

        adj = torch.from_numpy(adj).to(torch.float32).cuda()
        # print('adj.shape',adj.shape)
        # realize the multihead attention by concatenating the single head attention
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print('x.shape',x.shape)

        x = F.dropout(x, 0.5, training=self.training)  # 1160,64
        # print('x.shape',x.shape)
        x1 = self.out_att(x, adj)
        x1 = self.fkernel(x1)
        gat1 = F.elu(x1) + self.a_para * self.cos(x1)  # (1856,64)

        # gat1 = F.elu(self.out_att(x, adj))  # (1856,64)
        # print('gat1.shape',gat1.shape)
        x = rearrange(gat1, '(b n) c -> b n c', b=int(len(adj) / a_initial.shape[1]),
                      n=a_initial.shape[1])  # torch.Size([32, 116, 64])
        # print('x.last.shape',x.shape)
        return x, a_initial  # F.log_softmax(x, dim=1)