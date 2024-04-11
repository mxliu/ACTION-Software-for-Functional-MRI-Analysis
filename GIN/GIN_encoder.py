#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:38:35 2024

@author: qqw
"""


import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')
import os
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(
                torch.Tensor([[0.0]]))  # assumes that the adjacency matrix includes self-loop
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v
        v_combine = self.mlp(v_aggregate)
        return v_combine


class Module_1(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super(Module_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #  self.output_dim = output_dim

        self.percentile = Percentile()
        # 三个gcn层 (N,input_dim) -> (N,hidden_dim)
        self.gin1 = LayerGIN(input_dim, hidden_dim, hidden_dim)
        self.gin2 = LayerGIN(hidden_dim, hidden_dim, hidden_dim)
        # self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def _collate_adjacency(self, a):
        i_list = []
        v_list = []
        # torch.Size([4, 116, 116])
        for sample, _a in enumerate(a):
            # _a (116,116)
            # thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
            _i = _a.nonzero(as_tuple=False)
            _v = torch.ones(len(_i))
            _i += sample * a.shape[1]
            i_list.append(_i)
            v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)  # (2,121426)
        _v = torch.cat(v_list).to(a.device)
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0] * a.shape[1], a.shape[0] * a.shape[2]))

    def forward(self, X):
        fc_list = []
        for t in X:
            fc = corrcoef(t.T)
            fc_list.append(fc)
        fc1 = torch.stack(fc_list)
        fc1 = torch.nan_to_num(fc1)
        # node feature
        v = abs(fc1)
        v = torch.nan_to_num(v)
        v = rearrange(v, 'b n c -> (b n) c')
        v = v.to(torch.float32)
        v = v.cuda()
        a = self._collate_adjacency(fc1)
        # print(a.shape)#torch.Size([464, 464])
        x = F.relu(self.gin1(v, a))
        x = F.relu(self.gin2(x, a))
        x1 = rearrange(x, '(b n) c -> b n c', b=X.shape[0], n=X.shape[2])
        return x1


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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


class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input)  # find percentiles for flattened axis
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
        positions = percentiles * (input.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0 + d1).view(-1, *input_shape[1:])
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
        in_argsort = (in_argsort * input_shape[1] + cols_offsets).view(-1).long()
        floored = (
                floored[:, None] * input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
                ceiled[:, None] * input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input

