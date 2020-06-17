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
    def __init__(self, layers, n_features, n_classes, hidden_dims, dropout=0.5, gpu=False, flag=False):
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

    def forward(self, x, adjs):
        """
        修改意见：https://github.com/THUDM/cogdl/blob/master/cogdl/models/nn/pyg_gcn.py
        :param x:
        :param edge_index:
        :return:
        """
        device = torch.device('cuda' if self.gpu else 'cpu')

        # for i in range(self.layers - 1):
        #     nvtx_push(self.gpu, "layer" + str(i))
        #     x = self.convs[i](x, edge_index)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     nvtx_pop(self.gpu)
        #     log_memory(self.flag, device, 'layer' + str(i))

        # nvtx_push(self.gpu, "layer" + str(self.layers - 1))
        # x = self.convs[-1](x, edge_index)
        # nvtx_pop(self.gpu)
        # log_memory(self.flag, device, "layer" + str(self.layers - 1))
        # return F.log_softmax(x, dim=-1)
        for i, (edge_index, _, size) in enumerate(adjs):
            # x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i](x, edge_index, size=size[1])
            if i != self.layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                
        return x.log_softmax(dim=-1)

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



