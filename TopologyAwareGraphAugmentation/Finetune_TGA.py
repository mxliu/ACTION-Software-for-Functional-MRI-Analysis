
import warnings

warnings.filterwarnings('ignore')
import os
import torch.backends.cudnn as cudnn
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
import scipy.io
import model_finetune
import argparse
parser = argparse.ArgumentParser(description='PyTorch MDD Finetne')

parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-b1', '--batch-size1', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-b2', '--batch-size2', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # ？
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')  #

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')

# additional configs:
parser.add_argument('--pretrained', default='AA_1591checkpointHND_009.pth.tar', type=str,#AA_1591checkpointHND_009.pth.tar
                    help='path to simsiam pretrained checkpoint')

args = parser.parse_args()


def k_smallest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[::-1][:-k - 1:-1]

    return np.column_stack(np.unravel_index(idx, a.shape))


class MDD(object):
    def read_data(self):
        HC = scipy.io.loadmat('/home/qqw/Downloads/newfMRI_data/HC.mat')['A']
        ANI = scipy.io.loadmat('/home/qqw/Downloads/newfMRI_data/ANI.mat')['A']
        #
        import numpy as np
        alldata = np.concatenate((HC, ANI), axis=1)
        A = np.squeeze(alldata.T)  # (137,)
        adj_list = []
        fea_list = []
        for i in range(len(A)):  # 人数
            signal = A[i]  # (232,116)
            pc = np.corrcoef(signal.T)
            pc = np.nan_to_num(pc)
            fea_list.append(pc)
            pc_idx = k_smallest_index_argsort(pc, k=int(0.7 * len(pc) * len(pc)))
            for m, n in zip(pc_idx[:, 0], pc_idx[:, 1]):
                pc[m, n] = 0
            adj_list.append(pc)
        adj = np.array(adj_list)
        fea = np.array(fea_list)#using pc as node feature
        y2 = np.zeros(69)
        y3 = np.ones(68)
        y = np.concatenate((y2, y3), axis=0)
        return adj, fea, y

    def __init__(self):
        super(MDD, self).__init__()
        adj, fea, y = self.read_data()

        self.adj = torch.from_numpy(adj)
        self.fea = torch.from_numpy(fea)
        self.y = torch.from_numpy(y)
        self.n_samples = adj.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.adj[index], self.fea[index], self.y[index]


full_dataset = MDD()
import torch
import numpy as np
import random
seed = 0  # fold
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from sklearn.model_selection import KFold
k = 5
# kfold=KFold(n_splits=k,random_state=10,shuffle=True)
kfold = KFold(n_splits=k, random_state=seed, shuffle=True)
Acc2 = []
Sen2 = []
Spe2 = []
Bac2 = []
Ppv2 = []
Npv2 = []
Pre2 = []
Rec2 = []
F1_score2 = []
Auc2 = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(full_dataset)):
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    training_data_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=8, sampler=train_subsampler)  # 16,64
    testing_data_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=8, sampler=test_subsampler)


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
        degree = np.array(adjacency.sum(1))  #
        d_hat = sp.diags(np.power(degree, -0.5).flatten())  #
        L = d_hat.dot(adjacency).dot(d_hat).tocoo()  #
        indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
        values = torch.from_numpy(L.data.astype(np.float32))
        tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)

        return tensor_adjacency


    def global_max_pool(x, graph_indicator):
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
        num = graph_indicator.max().item() + 1

        return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


    def obtain_adjandfea(X):
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
        adj_nor = normalization(adj_csr).to(device)
        adj_nor = adj_nor.to(torch.float32)
        fc_list = []
        for i in range(len(X)):
            pc = np.corrcoef(X[i].cpu().T)
            pc = np.nan_to_num(pc)
            # pc = abs(pc)
            fc_list.append(pc)
        a = np.array(fc_list)  # (32, 116, 116)
        a_ = abs(a)
        a = torch.from_numpy(a)
        a_ = torch.from_numpy(a_)
        fea = rearrange(a, 'a b c-> (a b) c').to(device)
        # fea=torch.nan_to_num(fea)
        fea = fea.to(torch.float32)
        return adj_nor, fea

    def get_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'


    # device = get_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'DEVICE: {device}')


    model = model_finetune.Model(args.dim, args.pred_dim)
    model.to(device)
    print(model)
    #  Initialize weights with normal distribution (mean=0.0, std=0.01)
    model.predictor.L1.weight.data.normal_(mean=0.0, std=0.01)
    model.predictor.BN.weight.data.normal_(mean=0.0, std=0.01)
    model.predictor.L2.weight.data.normal_(mean=0.0, std=0.01)
    ## Initialize biases to zero
    #    model.predictor.L1.bias.data.zero_()
    model.predictor.BN.bias.data.zero_()
    model.predictor.L2.bias.data.zero_()

    if args.pretrained:
        if os.path.isfile(args.pretrained):  # os.path.isfile()：
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']  # in pretrain stage   'state_dict': Model.state_dict(),
            for k in list(state_dict.keys()):
                if k.startswith('first_encoder'):
                    # remove prefix
                    state_dict[k[len("first_"):]] = state_dict[k]
                    # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # print(set(msg.missing_keys))
            assert set(msg.missing_keys) == {"predictor.L1.weight", "predictor.BN.weight",
                                             "predictor.L2.weight", "predictor.BN.bias",
                                             "predictor.L2.bias", "predictor.BN.running_mean",
                                             "predictor.BN.num_batches_tracked",
                                             "predictor.BN.running_var"}  #

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    criterion = nn.CrossEntropyLoss()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)  #
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])  #
            optimizer.load_state_dict(checkpoint['optimizer'])  #
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    from einops import repeat, rearrange, reduce

    Acc = []
    Sen = []
    Spe = []
    Bac = []
    Ppv = []
    Npv = []
    Pre = []
    Rec = []
    F1_score = []
    Auc = []
    for epoch in range(args.start_epoch, args.epochs):
        # print(epoch)
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
        # training
        model.train()  # set the model to training mode
        for i, data in enumerate(training_data_loader):
            a, f, label = data  # torch.Size([32, 232, 116])
            a = a.to(device)
            f = f.to(device)
            label = label.to(device)
            # X = X.to(device)
            optimizer.zero_grad()
            z,outputs= model(a, f)
            batch_loss = criterion(outputs, label.long())
            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            batch_loss.backward()
            optimizer.step()
            train_acc += (train_pred.cpu() == label.cpu()).sum().item()
            train_loss += batch_loss.item()
        from einops import reduce, repeat

        Labels = []
        Test_pred = []
        Pre_score = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testing_data_loader):
                a, f, label = data  # torch.Size([32, 232, 116])
                a = a.to(device)
                f = f.to(device)
                label = label.to(device)
                Labels.append(label)
                z,output = model(a, f)
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

    acc, sen, spe, bac, ppv, npv, pre, rec, f1_score = calculate_metric(y_true, y_pred)
    # print("acc, sen, spe, bac, ppv, npv, pre, rec, f1_score",acc, sen, spe, bac, ppv, npv, pre, rec, f1_score)
    from sklearn import metrics

    fpr, tpr, threshold = metrics.roc_curve(y_true, PPre_score)
    auc = metrics.auc(fpr, tpr)
    Acc2.append(acc)
    Sen2.append(sen)
    Spe2.append(spe)
    Bac2.append(bac)
    Ppv2.append(ppv)
    Npv2.append(npv)
    Pre2.append(pre)
    Rec2.append(rec)
    F1_score2.append(f1_score)
    Auc2.append(auc)
print(type(Acc2), Acc2)
print(type(k), k)
k=5
avg_Acc = sum(Acc2) / k
avg_Acc = sum(Acc2) / k  #
print(avg_Acc)
print('Acc2std', np.std(Acc2, ddof=1))
avg_Sen = sum(Sen2) / k
print(avg_Sen)
print('Sen2std', np.std(Sen2, ddof=1))
avg_Spe = sum(Spe2) / k
print(avg_Spe)
print('Spe2std', np.std(Spe2, ddof=1))
avg_Bac = sum(Bac2) / k
print(avg_Bac)
print('Bac2std', np.std(Bac2, ddof=1))
avg_Ppv = sum(Ppv2) / k
print(avg_Ppv)
print('Ppv2std', np.std(Ppv2, ddof=1))
avg_Npv = sum(Npv2) / k
print(avg_Npv)
print('Npv2std', np.std(Npv2, ddof=1))
avg_Pre = sum(Pre2) / k
print(avg_Pre)
print('Pre2std', np.std(Pre2, ddof=1))
avg_Rec = sum(Rec2) / k
print(avg_Rec)
print('Rec2std', np.std(Rec2, ddof=1))
avg_F1_score = sum(F1_score2) / k
print(avg_F1_score)
print('F1_score2std', np.std(F1_score2, ddof=1))
avg_Auc = sum(Auc2) / k
print(avg_Auc)
print('Auc2std', np.std(Auc2, ddof=1))
