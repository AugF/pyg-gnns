import torch
from torch.nn import Parameter
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
    def __init__(self, gpu=False):
        super(MeanAggregate, self).__init__(aggr='mean', gpu=gpu)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GaANConv(MessagePassing):
    """
    GaAN layer
    """
    def __init__(self, in_channels, out_channels, d_a, d_v, d_m, heads,
                 bias=True, gpu=False):
        super(GaANConv, self).__init__(aggr='add', gpu=gpu)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.mid_units = d_a
        self.value_units = d_v
        self.max_pooling_units = d_m
        self.gpu = gpu
        device = torch.device('cuda' if gpu else 'cpu')

        self.weight_neigh = Parameter(torch.Tensor(in_channels, heads * d_a)).to(device)
        self.weight_edge = Parameter(torch.Tensor(1, heads, 2 * d_a)).to(device)
        self.weight_data = Parameter(torch.Tensor(in_channels, heads * d_v)).to(device)
        self.weight_att = Parameter(torch.Tensor(in_channels + heads * d_v, out_channels)).to(device)

        self.maxaggregate = MaxAggregate(gpu=gpu)
        self.meanaggregate = MeanAggregate(gpu=gpu)
        self.weight_max_polling = Parameter(torch.Tensor(in_channels, d_m)).to(device)
        self.weight_gate = Parameter(torch.Tensor(in_channels * 2 + d_m, heads)).to(device)

        if bias:
            self.bias = Parameter(torch.Tensor(self.heads, self.value_units)).to(device)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_neigh)
        glorot(self.weight_edge)
        glorot(self.weight_data)
        glorot(self.weight_att)
        glorot(self.weight_max_polling)
        glorot(self.weight_gate)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        # attentions = multi-head
        nvtx_push(self.gpu, "edge-cal_attentions")
        attentions = self.propagate(edge_index, size=size, x=x) # edge_cal
        nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "edge-cal_gateMax")
        gate_max = self.maxaggregate(torch.matmul(x, self.weight_max_polling), edge_index) # edge_cal
        nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "edge-cal_gateMean")
        gate_min = self.meanaggregate(x, edge_index)  # edge_cal
        nvtx_pop(self.gpu)

        nvtx_push(self.gpu, "vertex-cal")
        # gate = FC_theta_g(xi || max || mean)
        # ti = gate * multi-head
        output = torch.matmul(torch.cat([x, gate_max, gate_min], dim=-1), self.weight_gate).view(-1, self.heads, 1) # vertex_cal
        output = attentions * output

        # yi = FC_theta_o(xi || ti)
        output = torch.matmul(torch.cat([x, output.view(-1, self.heads * self.value_units)], dim=-1), self.weight_att) # vertex cal
        nvtx_pop(self.gpu)
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

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, d_a={}, d_v={}, d_m={}, heads={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.mid_units, self.value_units,
            self.max_pooling_units, self.heads
        )
