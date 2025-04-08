
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

#Ours
from data_source_R import HND
from data_source_R import WER

#Random
from data_source_R import RER
from data_source_R import RND

parser = argparse.ArgumentParser(description='PyTorch MDD Training')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
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

# simsiam specific configs:
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True 


train_dataset=HND()

def train(train_loader, Model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))


    Model.train()

    end = time.time()

    for i,(adj1, fea1, adj2, fea2) in enumerate(train_loader):
        data_time.update(time.time() - end)
        adj1=adj1.float().to(device)
        adj2 = adj2.float().to(device)
        fea1 = fea1.float().to(device)
        fea2 = fea2.float().to(device)
        p1, p2, z1, z2 = Model( fea1, fea2,adj1,adj2)

        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
              
        losses.update(loss.item(), fea1.size(0))   #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
Model = model_pretrain.SimSiam(args.dim, args.pred_dim)
Model.to(device)
print(Model)
criterion = nn.CosineSimilarity()

if args.fix_pred_lr:
    optim_params = [{'params': Model.first_encoder.parameters(), 'fix_lr': False},
                    {'params': Model.second_encoder.parameters(), 'fix_lr': False},
                    {'params': Model.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = Model.parameters()

optimizer = torch.optim.SGD(optim_params, args.lr,momentum=args.momentum,
                             weight_decay=args.weight_decay)


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



train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

training_loss = 0.0

losses = []
for epoch in range(args.start_epoch, args.epochs):
    
    adjust_learning_rate(optimizer, epoch, args)
    loss=train(train_loader, Model, criterion, optimizer, epoch, args)
    print('here',loss)
    losses.append(loss.cpu().detach().numpy() )
    if (epoch+1) % 10 == 0:
       save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': Model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='AA_1591checkpointHND_{:03d}.pth.tar'.format(epoch))


 









  


