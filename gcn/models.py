import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module
from gcn.layers import GCNConv

from inits import glorot

from utils import nvtx_push, nvtx_pop


class GCN(Module):
    """
    GCN layer
    dropout set: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims, dropout=0.5, gpu=False):
        super(GCN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.dropout = dropout
        self.gpu = gpu

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims))
        self.weight_out = Parameter(torch.Tensor(hidden_dims, n_classes))
        self.conv = torch.nn.ModuleList(
            [
                GCNConv(in_channels=hidden_dims, out_channels=hidden_dims, gpu=gpu, cached=True)
                for i in range(layers)
            ]
        )

        glorot(self.weight_in)
        glorot(self.weight_out)

    def forward(self, x, edge_index):
        nvtx_push(self.gpu, "input-transform")
        x = torch.spmm(x, self.weight_in)
        nvtx_pop(self.gpu)

        for i in range(self.layers):
            nvtx_push(self.gpu, "layer" + str(i))
            x = self.conv[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "output-transform")
        x = torch.matmul(x, self.weight_out)
        nvtx_pop(self.gpu)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, dropout={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.hidden_dims,
            self.dropout, self.gpu) + '\nLayer(conv->relu->dropout)\n' + str(self.conv)



