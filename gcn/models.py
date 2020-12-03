import torch
import sys

import torch.nn.functional as F
from torch.nn import Module
from gcn.layers import GCNConv
from utils import gcn_cluster_norm
from tqdm import tqdm


from utils import nvtx_push, nvtx_pop, log_memory

class GCN(Module):
    """
    GCN layer
    dropout set: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims, norm=None, dropout=0.5, gpu=False, device="cpu", flag=False, cluster_flag=False): # add adj
        super(GCN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.dropout = dropout
        self.gpu = gpu
        self.flag = flag
        self.device = device

        shapes = [n_features] + [hidden_dims] * (layers - 1) + [n_classes]
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(in_channels=shapes[layer], out_channels=shapes[layer + 1], gpu=gpu, cached=True)
                for layer in range(layers)
            ]
        )
        if norm is not None:
            self.norm = norm.to('cuda')
        self.cluster_flag = cluster_flag
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, x, adjs):
        """
        修改意见：https://github.com/THUDM/cogdl/blob/master/cogdl/models/nn/pyg_gcn.py
        :param x:
        :param edge_index:
        :return:
        """
        device = torch.device(self.device)
        
        if isinstance(adjs, list):
            for i, (edge_index, e_id, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            if  self.cluster_flag:
                norm = gcn_cluster_norm(adjs, x.size(0), None, False, x.dtype)
            else:
                norm = None
            for i in range(self.layers):
                nvtx_push(self.gpu, "layer" + str(i))
                x = self.convs[i](x, adjs, norm=norm)
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
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
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x = self.convs[i](x, edge_index, size=size[1], norm=self.norm[e_id])
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, dropout={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.hidden_dims,
            self.dropout, self.gpu) + '\nLayer(conv->relu->dropout)\n' + str(self.convs)



