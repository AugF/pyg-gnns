import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from inits import glorot, zeros

import torch.cuda.nvtx as nvtx

class MaxAggregate(MessagePassing):
    """
    max aggregate
    """
    def __init__(self):
        super(MaxAggregate, self).__init__(aggr="max")

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class MeanAggregate(MessagePassing):
    """
    mean aggregate
    """
    def __init__(self):
        super(MeanAggregate, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GaANConv(MessagePassing):
    """
    参考论文：GaAN
    """
    def __init__(self, in_channels, out_channels, d_a, d_v, d_m, heads,
                 bias=True):
        super(GaANConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.mid_units = d_a
        self.value_units = d_v
        self.max_pooling_units = d_m

        self.weight_neigh = Parameter(torch.Tensor(in_channels, heads * d_a))
        self.weight_edge = Parameter(torch.Tensor(1, heads, 2 * d_a))
        self.weight_data = Parameter(torch.Tensor(in_channels, heads * d_v))
        self.weight_att = Parameter(torch.Tensor(in_channels + heads * d_v, out_channels))

        self.maxaggregate = MaxAggregate()
        self.meanaggregate = MeanAggregate()
        self.weight_max_polling = Parameter(torch.Tensor(in_channels, d_m))
        self.weight_gate = Parameter(torch.Tensor(in_channels * 2 + d_m, heads))

        if bias:
            self.bias = Parameter(torch.Tensor(self.heads, self.value_units))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_neigh.data)
        glorot(self.weight_edge.data)
        glorot(self.weight_data.data)
        glorot(self.weight_att.data)
        glorot(self.weight_max_polling.data)
        glorot(self.weight_gate.data)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        # attentions = multi-head
        nvtx.range_push("edge-cal_attentions")
        attentions = self.propagate(edge_index, size=size, x=x) # edge_cal
        nvtx.range_pop()

        nvtx.range_push("edge-cal_gateMax")
        gate_max = self.maxaggregate(torch.matmul(x, self.weight_max_polling), edge_index) # edge_cal
        nvtx.range_pop()

        nvtx.range_push("edge-cal_gateMean")
        gate_min = self.meanaggregate(x, edge_index)  # edge_cal
        nvtx.range_pop()

        nvtx.range_push("vertex-cal")
        # gate = FC_theta_g(xi || max || mean)
        # ti = gate * multi-head
        output = torch.matmul(torch.cat([x, gate_max, gate_min], dim=-1), self.weight_gate).view(-1, self.heads, 1) # vertex_cal
        output = attentions * output

        # yi = FC_theta_o(xi || ti)
        output = torch.matmul(torch.cat([x, output.view(-1, self.heads * self.value_units)], dim=-1), self.weight_att) # vertex cal
        nvtx.range_pop()
        return output


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        data = torch.matmul(x_j, self.weight_data).view(-1, self.heads, self.value_units) # vertex cal
        x_i = torch.matmul(x_i, self.weight_neigh).view(-1, self.heads, self.mid_units)
        x_j = torch.matmul(x_j, self.weight_neigh).view(-1, self.heads, self.mid_units)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.weight_edge).sum(dim=-1)

        alpha = softmax(alpha, edge_index_i, size_i)

        output = data * alpha.view(-1, self.heads, 1)
        return output

    def update(self, aggr_out): # 注意到这里不是取平均，而是直接拼接
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
