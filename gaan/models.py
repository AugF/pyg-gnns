import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module

from gaan.layers import GaANConv
from inits import glorot
from utils import nvtx_push, nvtx_pop


class GaAN(Module):
    """
    GaAN model
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims,
                 heads, d_v, d_a, d_m, dropout=0.2, negative_slop=0.1, gpu=False):
        super(GaAN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.dropout, self.negative_slop = dropout, negative_slop
        self.gpu = gpu

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims))
        self.weight_out = Parameter(torch.Tensor(hidden_dims, n_classes))

        self.conv = [GaANConv(in_channels=hidden_dims,
                              out_channels=hidden_dims,
                              d_a=d_a, d_m=d_m, d_v=d_v, heads=heads,
                              gpu=gpu) for i in range(layers)]
        glorot(self.weight_in.data)
        glorot(self.weight_out.data)

    def forward(self, x, edge_index):
        nvtx_push(self.gpu, "input-transform")
        x = torch.matmul(x, self.weight_in)
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







