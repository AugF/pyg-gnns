import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module
from gat.layers import GATConv
from tqdm import tqdm

from inits import glorot
from utils import nvtx_push, nvtx_pop, log_memory


class GAT(Module):
    """
    GAT model
    dropout, negative_slop set: https://github.com/Diego999/pyGAT/blob/master/train.py
    """
    def __init__(self, layers, n_features, n_classes, head_dims,
                 heads, dropout=0.6, negative_slop=0.2, gpu=False, device="cpu", flag=False, sparse_flag=False):
        super(GAT, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.head_dims, self.heads = layers, head_dims, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.gpu = gpu
        self.device = device
        self.flag = flag
        self.sparse_flag = sparse_flag
        
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

    def forward(self, x, adjs):
        device = torch.device(self.device)
        
        if isinstance(adjs, list):
            for i, (edge_index, _, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                if not self.sparse_flag:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i]((x, x[:size[1]]), edge_index)
                if i != self.layers - 1:
                    x = F.elu(x)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            for i in range(self.layers):
                nvtx_push(self.gpu, "layer" + str(i))
                if not self.sparse_flag:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i](x, adjs)
                if i != self.layers - 1:
                    x = F.elu(x)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
                
        return x
    
    def inference(self, x_all, subgraph_loader):
        device = torch.device(self.device)
        
        pbar = tqdm(total=x_all.size(0) * self.layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x = F.dropout(x, p=self.dropout, training=self.training)
                # x_target = x[:size[1]]
                x = self.convs[i]((x, x[:size[1]]), edge_index)
                if i != self.layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
    
    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, head_dims={}, heads={}' \
               ', dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.head_dims,
            self.heads, self.dropout, self.negative_slop, self.gpu) + '\nLayer(dropout->conv->elu)\n' + str(self.convs)



