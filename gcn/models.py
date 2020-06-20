import torch

import torch.nn.functional as F
from torch.nn import Module
from gcn.layers import GCNConv
from tqdm import tqdm


from utils import nvtx_push, nvtx_pop, log_memory

class GCN(Module):
    """
    GCN layer
    dropout set: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims, adj, dropout=0.5, gpu=False, flag=False): # add adj
        super(GCN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.dropout = dropout
        self.gpu = gpu
        self.flag = flag

        shapes = [n_features] + [hidden_dims] * (layers - 1) + [n_classes]
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(in_channels=shapes[layer], out_channels=shapes[layer + 1], gpu=gpu, cached=True)
                for layer in range(layers)
            ]
        )
        self.adj = adj

    def get_norm(self, edge_index):
        edge_weight = []
        adj = self.adj.todense()
        for i in range(edge_index.shape[1]):
            e = edge_index[:, i]
            edge_weight.append(adj[e[0], e[1]])
        return torch.FloatTensor(edge_weight)
    
    def forward(self, x, adjs):
        """
        修改意见：https://github.com/THUDM/cogdl/blob/master/cogdl/models/nn/pyg_gcn.py
        :param x:
        :param edge_index:
        :return:
        """
        device = torch.device('cuda' if self.gpu else 'cpu')
        
        if isinstance(adjs, list):
            for i, (edge_index, _, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                norm = self.get_norm(edge_index).to(device)
                x = self.convs[i](x, edge_index, size=size[1], norm=norm)
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
        else:
            for i in range(self.layers):
                nvtx_push(self.gpu, "layer" + str(i))
                norm = self.get_norm(adjs).to(device)
                x = self.convs[i](x, adjs, norm=norm)
                if i != self.layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                nvtx_pop(self.gpu)
                
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader):
        device = torch.device('cuda' if self.gpu else 'cpu')
        
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
                # x_target = x[:size[1]]
                x = self.convs[i](x, edge_index, size=size[1])
                if i != self.layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, dropout={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.hidden_dims,
            self.dropout, self.gpu) + '\nLayer(conv->relu->dropout)\n' + str(self.convs)



