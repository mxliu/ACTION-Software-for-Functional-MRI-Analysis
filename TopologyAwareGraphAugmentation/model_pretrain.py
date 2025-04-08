

import torch.nn as nn
from gnn_encoder import Module_1

from einops import repeat,rearrange,reduce

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=64, pred_dim=32):

        super(SimSiam, self).__init__()

        self.first_encoder=Module_1(116,dim)

      


        self.predictor = nn.Sequential()
        self.predictor.add_module('L1',nn.Linear(dim, pred_dim,bias=False)),
        self.predictor.add_module('BN',nn.BatchNorm1d(pred_dim)),
        self.predictor.add_module('RL',nn.ReLU(inplace=True)), # hidden layer
        self.predictor.add_module('L2',  nn.Linear(pred_dim, dim)) # output layer
 
    def forward(self, x1, x2,A1,A2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        z1 = self. first_encoder(A1,x1)
        z2 = self. first_encoder(A2,x2)#
        z1=reduce(z1,'b n c ->b c','mean')
        z2 = reduce(z2, 'b n c ->b c', 'mean')

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()