import torch
import sys
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from message_passing import MessagePassing
from inits import glorot, zeros
from utils import nvtx_push, nvtx_pop


class MaxAggregate(MessagePassing):
    """
    max aggregate
    """
    def __init__(self, gpu=False):
        super(MaxAggregate, self).__init__(aggr="max", gpu=gpu)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class MeanAggregate(MessagePassing):
    """
    mean aggregate
    """
    def __init__(self, gpu=False):
        super(MeanAggregate, self).__init__(aggr='mean', gpu=gpu)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GaANConv(MessagePassing):
    """
    GaAN layer
    """
    def __init__(self, in_channels, out_channels, d_a, d_v, d_m, heads,
                 gpu=False):
        super(GaANConv, self).__init__(aggr='add', gpu=gpu)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.mid_units = d_a
        self.value_units = d_v
        self.max_pooling_units = d_m
        self.gpu = gpu

        self.lin_neigh = torch.nn.Linear(in_channels, heads * d_a)
        self.lin_data = torch.nn.Linear(in_channels, heads * d_v)
        
        self.maxaggregate = MaxAggregate(gpu=gpu)
        self.meanaggregate = MeanAggregate(gpu=gpu)
        self.lin_gate_max = torch.nn.Linear(in_channels, d_m)
        self.lin_gate = torch.nn.Linear(in_channels * 2 + d_m, heads)
        self.lin_out = torch.nn.Linear(in_channels + heads * d_v, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_neigh.reset_parameters()
        self.lin_data.reset_parameters()
        self.lin_gate_max.reset_parameters()
        self.lin_gate.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, x, edge_index, size=None):
        # attentions = multi-head
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                            num_nodes=x.size(self.node_dim))
                   
        nvtx_push(self.gpu, "edge-cal_attentions")
        if size is not None:
            attentions = self.propagate(edge_index, x=(x, x[:size])) # edge_cal
        else:
            attentions = self.propagate(edge_index, x=x)
        nvtx_pop(self.gpu)
        
        nvtx_push(self.gpu, "edge-cal_gateMax")
        ht = self.lin_gate_max(x)
        if size is not None:
            gate_max = self.maxaggregate((ht, ht[:size]), edge_index) # edge_cal
        else:
            gate_max = self.maxaggregate(ht, edge_index) # edge_cal
        nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "edge-cal_gateMean")
        if size is not None:
            gate_mean = self.meanaggregate((x, x[:size]), edge_index)  # edge_cal
        else:
            gate_mean = self.meanaggregate(x, edge_index)  # edge_cal
        nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "vertex-cal")
        
        if size is not None:
            x = x[:size]
        # gate = FC_theta_g(xi || max || mean)
        # ti = gate * multi-head
        output = self.lin_gate(torch.cat([x, gate_max, gate_mean], dim=-1)).view(-1, self.heads, 1)
        output = attentions * output

        # yi = FC_theta_o(xi || ti)
        output = self.lin_out(torch.cat([x, output.view(-1, self.heads * self.value_units)], dim=-1)) # vertex cal
        nvtx_pop(self.gpu)
        return output

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        data = self.lin_data(x_j).view(-1, self.heads, self.value_units)
        x_i = self.lin_neigh(x_i).view(-1, self.heads, 1, self.mid_units)
        x_j = self.lin_neigh(x_j).view(-1, self.heads, self.mid_units, 1)

        alpha = torch.matmul(x_i, x_j).squeeze_()
        alpha = softmax(alpha, edge_index_i, size_i)
        data = F.leaky_relu(data, 0.1)
        output = data * alpha.view(-1, self.heads, 1)
        return output

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, d_a={}, d_v={}, d_m={}, heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.mid_units, self.value_units,
            self.max_pooling_units, self.heads
        )
