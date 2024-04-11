#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:44:36 2024

@author: qqw

"""

import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim
import torch.backends.cudnn as cudnn
import math
import time
import shutil
import model_pretrain
import warnings
warnings.simplefilter('ignore')
import pickle
import numpy as np
import torch
import scipy
import torch.nn as nn
#parser.add_argument(xxx)
#

from einops import repeat, rearrange, reduce
import torch
import numpy as np
####################################Hyper-Parameters####################################
parser = argparse.ArgumentParser(description='PyTorch MDD Training')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

################### Here, you can select a specific encoder!###################
parser.add_argument('--encoder-type',
                    choices=['BrainGNN'],
                    default='BrainGNN',  # Change the default
                    help="Choose the type of encoder")

args = parser.parse_args()

#
Module_1 = getattr(__import__('{}_encoder'.format(args.encoder_type)), 'Module_1')
print(Module_1)



if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

####################################Data Loading####################################
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import numpy as np
import os.path as osp
from imports.read_abide_stats_parall import read_data
from torch_geometric.data import DataLoader
class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ABIDEDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root, 'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices = read_data(self.raw_dir)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


full_dataset1 = ABIDEDataset('BrainGNN/V1', 'ABIDE')

full_dataset2 = ABIDEDataset('BrainGNN/V2', 'ABIDE')
train_loader1 = DataLoader(
    full_dataset1, batch_size=args.batch_size, shuffle=False)

train_loader2 = DataLoader(
    full_dataset2, batch_size=args.batch_size, shuffle=False)
#     print("Data:", data)
# for i,(data1, data2) in enumerate( zip(train_loader1, train_loader1)):
#     print(i)
#   #  print(data1.x.shape)  # torch.Size([116, 116])
#     print(data1.x)  # torch.Size([1])
#     print(data2.x)
    # print(data1.edge_index.shape)  # torch.Size([2, 13340])
   # print(data1.batch.shape)  # torch.Size([116])
   # print(data1.edge_attr.shape)  # torch.Size([13340, 1])
   # print(data1.pos.shape)  # torch.Size([116, 116])
###################Contrastive Learning Framework#############

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, dim, pred_dim):
        super(SimSiam, self).__init__()

        self.first_encoder = Module_1(116, 0.5)
        self.predictor = nn.Sequential()
        self.predictor.add_module('L1', nn.Linear(dim, pred_dim, bias=False)),
        self.predictor.add_module('BN', nn.BatchNorm1d(pred_dim)),
        self.predictor.add_module('RL', nn.ReLU(inplace=True)),  # hidden layer
        self.predictor.add_module('L2', nn.Linear(pred_dim, dim))  # output layer

    def forward(self, data1,data2):
    #def forward(self, x1, x2, A1, A2):
        """
        Input:
            data1: first views of signals
            data2: second views of signals
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        z1, v1_w1, v1_w2, v1_s1, v1_s2 = self.first_encoder(data1.x, data1.edge_index, data1.batch, data1.edge_attr, data1.pos)  # 16,104,64# NxC
        z2, v2_w1, v2_w2, v2_s1, v2_s2 = self.first_encoder(data2.x, data2.edge_index, data2.batch, data2.edge_attr, data2.pos)  # (nbatch,64,1,1) # NxC#
       # z1 = reduce(z1, 'b n c ->b c', 'mean')
       # z2 = reduce(z2, 'b n c ->b c', 'mean')
        p1 = self.predictor(z1)  # nbatch,64)
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach(), v1_w1, v1_w2, v1_s1, v1_s2,v2_w1, v2_w2, v2_s1, v2_s2

EPS = 1e-10
def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - torch.log(
        1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    return res

def train(train_loader1,train_loader2, Model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader1),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    Model.train()

    end = time.time()
    for i,(data1, data2) in enumerate( zip(train_loader1, train_loader1)):
        # Assuming 'y' is a tensor containing node labels
        #print(data1['y'])  # Accessing node labels assuming they are stored in 'y'
       # print(data2['y'])  # Accessing node labels assuming they are stored in 'y'

        # for i, (data1,data2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data1 = data1.to(device)
        data2 = data2.to(device)

        p1, p2, z1, z2, v1_w1, v1_w2, v1_s1, v1_s2,v2_w1, v2_w2, v2_s1, v2_s2  = Model(data1,data2)
        v1_loss_p1 = (torch.norm(v1_w1, p=2) - 1) ** 2
        v1_loss_p2 = (torch.norm(v1_w2, p=2) - 1) ** 2
        v1_loss_tpk1 = topk_loss(v1_s1, 0.5)
        v1_loss_tpk2 = topk_loss(v1_s2,  0.5)
        v2_loss_p1 = (torch.norm(v2_w1, p=2) - 1) ** 2
        v2_loss_p2 = (torch.norm(v2_w2, p=2) - 1) ** 2
        v2_loss_tpk1 = topk_loss(v2_s1, 0.5)
        v2_loss_tpk2 = topk_loss(v2_s2,  0.5)


        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5+ 0.1 * v1_loss_p1 + 0.1 * v1_loss_p2 \
                       + 0.1 * v1_loss_tpk1 + 0.1 * v1_loss_tpk2
        + 0.1 * v2_loss_p1 + 0.1 * v2_loss_p2 \
        + 0.1 * v2_loss_tpk1 + 0.1 * v2_loss_tpk2

        losses.update(loss.item(), data1.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    print(loss)
    return loss


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = args.lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] = cur_lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Model = SimSiam(args.dim, args.pred_dim)#model_pretrain.
Model.to(device)
print(Model)
criterion = nn.CosineSimilarity()

if args.fix_pred_lr:
    optim_params = [{'params': Model.first_encoder.parameters(), 'fix_lr': False},
                    {'params': Model.second_encoder.parameters(), 'fix_lr': False},
                    {'params': Model.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = Model.parameters()

optimizer = torch.optim.SGD(optim_params, args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(optim_params,args.lr,weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        Model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

training_loss = 0.0

losses = []
for epoch in range(args.start_epoch, args.epochs):

    adjust_learning_rate(optimizer, epoch, args)
    loss = train(train_loader1, train_loader2,Model, criterion, optimizer, epoch, args)
    print('here', loss)
    losses.append(loss.cpu().detach().numpy())
   # if epoch == 49:
    if (epoch+1) % 10 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': Model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='Pretrained_{}_{:03d}.pth.tar'.format(args.encoder_type,epoch))


import matplotlib.pyplot as plt
# Plot the loss over epochs
plt.plot(range(1, args.epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('{}.png'.format(args.encoder_type), dpi=300)









