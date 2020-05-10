import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module
from gaan.layers import GaANConv
from inits import glorot

import torch.cuda.nvtx as nvtx
import sys
sys.path.append("/home/wangzhaokang/wangyunpan/pyg-gnns")

class GaAN(Module):
    """

    """
    def __init__(self, layers, n_features, n_classes, hidden_dims,
                 heads, d_v, d_a, d_m, dropout=0.2, negative_slop=0.1):
        super(GaAN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.dropout, self.negative_slop = dropout, negative_slop

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims))
        self.weight_out = Parameter(torch.Tensor(hidden_dims, n_classes))

        self.conv = [GaANConv(hidden_dims, hidden_dims, d_a, d_v, d_m, heads) for i in range(layers)]
        glorot(self.weight_in.data)
        glorot(self.weight_out.data)

    def forward(self, x, edge_index):
        nvtx.range_push("input-transform")
        x = torch.matmul(x, self.weight_in)
        nvtx.range_pop()

        for i in range(self.layers):
            nvtx.range_push("layer" + str(i))
            x = F.leaky_relu(self.conv[i](x, edge_index), self.negative_slop)
            x = F.dropout(x, p=self.dropout, training=self.training)
            nvtx.range_pop()

        nvtx.range_push("output-transform")
        x = torch.matmul(x, self.weight_out)
        nvtx.range_pop()
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    """
    test models by cora
    """
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    import os.path as osp
    import numpy as np

    # 0. set manual seed
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    # 1. load data
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    # 2. model + Adam
    # 2.1 set to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = GaAN(
        layers=2,
        n_features=dataset.num_features, n_classes=dataset.num_classes, hidden_dims=128,
        heads=8, d_v=16, d_a=24, d_m=64
    ).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 3. train + test
    def train():
        nvtx.range_push("forward")
        model.train()
        loss = F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        nvtx.range_pop()
        nvtx.range_push("backward")
        loss.backward()
        optimizer.step()
        nvtx.range_pop()

    def test():
        model.eval()
        logits, accs = model(data.x, data.edge_index), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    def main():
        for epoch in range(11):
            nvtx.range_push("epoch " + epoch)
            nvtx.range_push("train")
            train()
            nvtx.range_pop()
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, *test()))
            nvtx.range_pop()

    with torch.cuda.profiler.profile():
        main() # warmip cuda memory allocator and profiler
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            main()




