import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module
from gat.layers import GATConv

from inits import glorot
from utils import nvtx_push, nvtx_pop


class GAT(Module):
    """
    GAT model
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims,
                 heads, dropout=0.2, negative_slop=0.1, gpu=False):
        super(GAT, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims, self.heads = layers, hidden_dims, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.gpu = gpu

        device = torch.device('cuda' if gpu else 'cpu')

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims * heads)).to(device)
        self.weight_out = Parameter(torch.Tensor(hidden_dims * heads, n_classes)).to(device)
        self.dropout = dropout
        self.conv = [GATConv(in_channels=hidden_dims * heads,
                             out_channels=hidden_dims,
                             heads=heads,
                             dropout=dropout,
                             negative_slope=negative_slop,
                             gpu=gpu) for i in range(layers)]
        glorot(self.weight_in.data)
        glorot(self.weight_out.data)

    def forward(self, x, edge_index):
        nvtx_push(self.gpu, "input-transform")
        x = torch.matmul(x, self.weight_in)
        nvtx_pop(self.gpu)

        for i in range(self.layers):
            nvtx_push(self.gpu, "layer" + str(i))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "output-transform")
        x = torch.matmul(x, self.weight_out)
        nvtx_pop(self.gpu)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, heads={}' \
               ', dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.hidden_dims,
            self.heads, self.dropout, self.negative_slop, self.gpu) + '\n' + str(self.conv)



