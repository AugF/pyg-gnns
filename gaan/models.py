import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module

from gaan.layers import GaANConv
from inits import glorot
from utils import nvtx_push, nvtx_pop, log_memory


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

        shapes = [n_features] + [hidden_dims] * (layers - 1) + [n_classes]
        self.convs = torch.nn.ModuleList(
            [
                GaANConv(in_channels=shapes[layer], out_channels=shapes[layer + 1],
                         d_a=d_a, d_m=d_m, d_v=d_v, heads=heads, gpu=gpu)
                for layer in range(layers)
            ]
        )

    def forward(self, x, edge_index):
        device = torch.device('cuda' if self.gpu else 'cpu')

        for i in range(self.layers - 1):
            nvtx_push(self.gpu, "layer" + str(i))
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x, self.negative_slop)
            x = F.dropout(x, p=self.dropout, training=self.training)
            nvtx_pop(self.gpu)
            log_memory(device, 'layer' + str(i))

        nvtx_push(self.gpu, "layer" + str(self.layers - 1))
        x = self.convs[-1](x, edge_index)
        nvtx_pop(self.gpu)
        log_memory(device, "layer" + str(self.layers - 1))
        return x

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, heads={},' \
               'd_v={}, d_a={}, d_m={}, dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes,
            self.hidden_dims, self.heads, self.d_v, self.d_a, self.d_m, self.dropout,
            self.negative_slop, self.gpu) + '\nLayer(conv->leaky_relu->dropout)\n' + str(self.convs[0])







