import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module

from gaan.layers import GaANConv
from inits import glorot
from utils import nvtx_push, nvtx_pop


class GaAN(Module):
    """
    GaAN model
    dropout, negative_slop set: GaAN: Gated attention networks for learning on large and spatiotemporal graphs 5.3
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims,
                 heads, d_v, d_a, d_m, dropout=0.1, negative_slop=0.1, gpu=False):
        super(GaAN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims, self.heads = layers, hidden_dims, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.d_v, self.d_a, self.d_m = d_v, d_a, d_m
        self.gpu = gpu

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims))
        self.weight_out = Parameter(torch.Tensor(hidden_dims, n_classes))

        self.conv = torch.nn.ModuleList(
            [
                GaANConv(in_channels=hidden_dims, out_channels=hidden_dims,
                         d_a=d_a, d_m=d_m, d_v=d_v, heads=heads, gpu=gpu)
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
            x = F.leaky_relu(x, self.negative_slop)
            x = F.dropout(x, p=self.dropout, training=self.training)
            nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "output-transform")
        x = torch.matmul(x, self.weight_out)
        nvtx_pop(self.gpu)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, heads={},' \
               'd_v={}, d_a={}, d_m={}, dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes,
            self.hidden_dims, self.heads, self.d_v, self.d_a, self.d_m, self.dropout,
            self.negative_slop, self.gpu) + '\nLayer(conv->leaky_relu->dropout)\n' + str(self.conv)







