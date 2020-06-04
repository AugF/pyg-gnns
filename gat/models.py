import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module
from gat.layers import GATConv

from inits import glorot
from utils import nvtx_push, nvtx_pop


class GAT(Module):
    """
    GAT model
    dropout, negative_slop set: https://github.com/Diego999/pyGAT/blob/master/train.py
    """
    def __init__(self, layers, n_features, n_classes, head_dims,
                 heads, dropout=0.6, negative_slop=0.2, gpu=False):
        super(GAT, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.head_dims, self.heads = layers, head_dims, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.gpu = gpu

        self.dropout = dropout
        self.conv_in = GATConv(in_channels=n_features, out_channels=head_dims, heads=heads, dropout=dropout)

        in_shapes = [n_features] + [head_dims * heads] * (layers - 1)
        out_shapes = [head_dims] * (layers - 1) + [n_classes]
        head_shape = [heads] * (layers - 1) + [1]
        self.convs = torch.nn.ModuleList(
            [
                GATConv(in_channels=in_shapes[layer], out_channels=out_shapes[layer],
                        heads=head_shape[layer], dropout=dropout, negative_slope=negative_slop,
                        gpu=gpu)
                for layer in range(layers)
             ]
        )

        self.conv_out = GATConv(in_channels=heads * head_dims, out_channels=n_classes, heads=1, dropout=dropout)

    def forward(self, x, edge_index):
        for i in range(self.layers - 1):
            nvtx_push(self.gpu, "layer" + str(i))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "layer" + str(self.layers - 1))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        nvtx_pop(self.gpu)
        return x

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, head_dims={}, heads={}' \
               ', dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.head_dims,
            self.heads, self.dropout, self.negative_slop, self.gpu) + '\nLayer(dropout->conv->elu)\n' + str(self.convs)



