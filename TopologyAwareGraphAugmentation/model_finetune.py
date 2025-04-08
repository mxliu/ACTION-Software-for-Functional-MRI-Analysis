

import torch.nn as nn

from gnn_encoder import Module_1
import torch.nn.functional as F

from einops import repeat,rearrange,reduce

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

       
        self.encoder=Module_1(116,dim)
        
        # build a 2-layer predictor
        self.predictor = nn.Sequential()
        self.predictor.add_module('L1',nn.Linear(dim, pred_dim,bias=False)),
        self.predictor.add_module('BN',nn.BatchNorm1d(pred_dim)),
        self.predictor.add_module('RL',nn.ReLU(inplace=True)), 
        self.predictor.add_module('L2',nn.Linear(pred_dim, 2)) # output layer
        #self.sig = nn.Sigmoid()
        
    def forward(self, A,x):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        z= self.encoder(A,x)
        z=reduce(z,'b n c ->b c','mean')
        p = self.predictor(z)# NxC
       
        output = F.softmax(p, dim=1) #
    
        return z,output