import torch
from torch import Tensor
from torch.nn import Parameter as Param
from message_passing import MessagePassing
import torch.nn.functional as F

from inits import uniform

from utils import nvtx_push, nvtx_pop, log_memory


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 gpu=False, flag=False):
        super(GatedGraphConv, self).__init__(aggr=aggr, gpu=gpu)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gpu = gpu
        self.flag = flag

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias) # gru是共享权重的

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight) # uniform
        self.rnn.reset_parameters()

    def forward(self, x, adjs, edge_weight=None, size=None):
        """"""
        device = torch.device('cuda' if self.gpu else 'cpu')
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        if isinstance(adjs, list):
            for i, (edge_index, _, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                nvtx_push(self.gpu, "vertex-cal_1")
                m = torch.matmul(h, self.weight[i]) # vertex cal
                nvtx_pop(self.gpu)  
                nvtx_push(self.gpu, "edge-cal")
                m = self.propagate(edge_index, x=(m, m[:size[1]]), edge_weight=edge_weight) # edge cal
                nvtx_pop(self.gpu)
                nvtx_push(self.gpu, "vertex-cal_2")
                h = self.rnn(m, h[:size[1]]) # vertex cal todo: 这里也有改变
                nvtx_pop(self.gpu)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            for i in range(self.num_layers):
                nvtx_push(self.gpu, "layer" + str(i))
                nvtx_push(self.gpu, "vertex-cal_1")
                m = torch.matmul(h, self.weight[i])
                nvtx_pop(self.gpu)  
                nvtx_push(self.gpu, "edge-cal")
                m = self.propagate(adjs, x=m, edge_weight=edge_weight)
                nvtx_pop(self.gpu)
                nvtx_push(self.gpu, "vertex-cal_2")
                h = self.rnn(m, h)
                nvtx_pop(self.gpu)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        return h
    

    def inference(self, x_all, subgraph_loader, pbar):
        device = torch.device('cuda' if self.gpu else 'cpu')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                
                # GRU单元
                m = torch.matmul(x, self.weight[i]) # vertex cal
                m = self.propagate(edge_index, x=(m, m[:size[1]]), edge_weight=None) # edge cal
                x = self.rnn(m, x[:size[1]]) # vertex cal todo: 这里也有改变
                
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        return x_all
    
    
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={}, aggr={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers, self.aggr)


