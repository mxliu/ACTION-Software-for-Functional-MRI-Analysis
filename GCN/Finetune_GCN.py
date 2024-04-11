#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:58:05 2024

@author: qqw
"""
#Note that please input your to-be-analyzed data in the class named Data, including bold with shape of (nsub,nlength,nroi) and label with the shape of (nsub,)
import warnings
warnings.filterwarnings('ignore')
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.init as init
#import model_finetune
import torch
import numpy as np
import random
from sklearn.model_selection import KFold
import argparse
import scipy

##############################Parameter Setting ##############################
parser = argparse.ArgumentParser(description='PyTorch Finetne')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batchsize', default=16, type=int, metavar='N',
                    help='the batch size to use for training and testing')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # ？
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')
##############################here, please input the path of pretrained model ##############################
# parser.add_argument('--pretrained', default='10_000_GCN_039.pth.tar', type=str,
#                     help='path to simsiam pretrained checkpoint')

parser.add_argument('--encoder-type',
                    choices=['GCN', 'GIN', 'GAT','BrainNetCNN','Transformer'],
                    default='GCN',  # Change the default to 'gat'
                    help="Choose the type of GNN encoder: 'gcn', 'gin', or 'gat' (default: 'gat')")


args = parser.parse_args()

seed=1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

####################################Bold Signal Augmentation####################################
class Data(object):
    def read_data(self):
        #########This is demo data###########
        bold = np.random.random((137, 230, 116))
        # bold shape (nsub,nlength,nroi) ROI=116
        y = np.random.randint(2, size=137) #y is label whose shape is (nsub,)
        print(y.shape)
        return bold,y

    def __init__(self):
        super(Data, self).__init__()
        bold,y= self.read_data()

        self.bold = torch.from_numpy(bold)
        self.y = torch.from_numpy(y)
        self.n_samples = bold.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.bold[index],self.y[index]

################################ functions for model finetuning functions################################

import torch.nn as nn
Module_1 = getattr(__import__('{}_encoder'.format(args.encoder_type)), 'Module_1')
print(Module_1)
import torch.nn.functional as F

from einops import repeat, rearrange, reduce


class Model(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, dim, pred_dim):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512) 这样设置维数与原设定相反，原来是两头大中间小
        """
        super(Model, self).__init__()
        self.encoder = Module_1(116, dim)
        self.predictor = nn.Sequential()
        self.predictor.add_module('L1', nn.Linear(dim, pred_dim, bias=False)),
        self.predictor.add_module('BN', nn.BatchNorm1d(pred_dim)),
        self.predictor.add_module('RL', nn.ReLU(inplace=True)),
        self.predictor.add_module('L2', nn.Linear(pred_dim, 2))  # output layer
        # self.sig = nn.Sigmoid()

    def forward(self, data):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        z = self.encoder(data)
        z = reduce(z, 'b n c ->b c', 'mean')  # z is tensor(8,64), which is graph representation generated by GNN
        p = self.predictor(z)  # NxC

        output = F.softmax(p, dim=1)  #

        return z, output


full_dataset = Data()
k = 5
kfold = KFold(n_splits=k, random_state=seed, shuffle=True)
Acc2 = []
Sen2 = []
Spe2 = []
Bac2 = []
Pre2 = []
F1_score2 = []
Auc2 = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(full_dataset)):
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    training_data_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batchsize, sampler=train_subsampler)  # 16,64
    testing_data_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batchsize, sampler=test_subsampler)
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
        pre = TP / float(TP + FP)
        rec = TP / float(TP + FN)
        f1_score = 2 * pre * rec / (pre + rec)
        return acc, sen, spe, bac, pre, f1_score

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'DEVICE: {device}')
    model =Model(args.dim, args.pred_dim)  # args.pred_dim
    model.to(device)
    print(model)
    #  Initialize weights with normal distribution (mean=0.0, std=0.01)
    model.predictor.L1.weight.data.normal_(mean=0.0, std=0.01)
    model.predictor.BN.weight.data.normal_(mean=0.0, std=0.01)
    model.predictor.L2.weight.data.normal_(mean=0.0, std=0.01)
    ## Initialize biases to zero
    model.predictor.BN.bias.data.zero_()
    model.predictor.L2.bias.data.zero_()

    ##############################To load the pretrained model ##############################
    pretrained='Pretrained_{}_019.pth.tar'.format(args.encoder_type)
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")  # load the checkpoint file
            state_dict = checkpoint['state_dict']  # Retrieve the state dictionary from the loaded checkpoint.
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('first_encoder'):
                    # remove prefix
                    state_dict[k[len("first_"):]] = state_dict[k]
                    # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)  # Load the modified state dictionary into the model.
            # print(set(msg.missing_keys))
            assert set(msg.missing_keys) == {"predictor.L1.weight", "predictor.BN.weight",
                                             "predictor.L2.weight", "predictor.BN.bias",
                                             "predictor.L2.bias", "predictor.BN.running_mean",
                                             "predictor.BN.num_batches_tracked",
                                             "predictor.BN.running_var"}

            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

    criterion = nn.CrossEntropyLoss()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    cudnn.benchmark = True
    ###This code is used for freeze the encoder
    # for param in  model.encoder.parameters():
    #     param.requires_grad = False
    grad = any(param.requires_grad for param in model.encoder.parameters())
    print(grad)
    ##############################Model Finetuning ##############################
    for epoch in range(args.start_epoch, args.epochs):
        # print(epoch)
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
        # training
        model.train()
        for i, data in enumerate(training_data_loader):
            bold, label = data
            bold = bold.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            z, outputs = model(bold)
            batch_loss = criterion(outputs, label.long())
            _, train_pred = torch.max(outputs, 1)
            batch_loss.backward()
            optimizer.step()
            train_acc += (train_pred.cpu() == label.cpu()).sum().item()
            train_loss += batch_loss.item()


        Labels = []
        Test_pred = []
        Pre_score = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testing_data_loader):
                bold, label = data  # torch.Size([32, 232, 116])
                bold = bold.to(device)
                # f = f.to(device)
                label = label.to(device)
                Labels.append(label)
                z, output = model(bold)
                batch_loss = criterion(output, label.long())
                pre_score = output[:, 1]
                Pre_score.append(pre_score)
                _, test_pred = torch.max(output, 1)
                Test_pred.append(test_pred)
                test_acc += (
                        test_pred.cpu() == label.cpu()).sum().item()  # get the index of the class with the highest probability
                test_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, args.epochs, train_acc / len(train_idx), train_loss / len(training_data_loader),
                test_acc / len(test_idx), test_loss / len(testing_data_loader)))

            y_true = torch.cat(Labels, -1).cpu()
            y_pred = torch.cat(Test_pred, -1).cpu()
            PPre_score = torch.cat(Pre_score, -1).cpu()

    acc, sen, spe, bac, pre, f1_score = calculate_metric(y_true, y_pred)
    from sklearn import metrics

    fpr, tpr, threshold = metrics.roc_curve(y_true, PPre_score)
    auc = metrics.auc(fpr, tpr)
    Acc2.append(acc)
    Sen2.append(sen)
    Spe2.append(spe)
    Bac2.append(bac)
    Pre2.append(pre)
    F1_score2.append(f1_score)
    Auc2.append(auc)

k = 5
avg_Acc = sum(Acc2) / k
print('Acc mean',avg_Acc)
print('Acc std', np.std(Acc2, ddof=1))
avg_Sen = sum(Sen2) / k
print('Sen mean',avg_Sen)
print('Sen std', np.std(Sen2, ddof=1))
avg_Spe = sum(Spe2) / k
print('Spe mean',avg_Spe)
print('Spe std', np.std(Spe2, ddof=1))
avg_Bac = sum(Bac2) / k
print('Bac mean',avg_Bac)
print('Bac std', np.std(Bac2, ddof=1))
avg_Pre = sum(Pre2) / k
print('Pre mean',avg_Pre)
print('Pre std', np.std(Pre2, ddof=1))
avg_F1_score = sum(F1_score2) / k
print('F1 mean',avg_F1_score)
print('F1_score std', np.std(F1_score2, ddof=1))
avg_Auc = sum(Auc2) / k
print('Auc mean',avg_Auc)
print('Auc std', np.std(Auc2, ddof=1))
