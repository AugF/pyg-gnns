import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torch
import numpy as np
import torch.nn.functional as F
from gaan.layers import GaANConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GaANConv(in_channels=dataset.num_features, out_channels=256,
                              heads=8, d_v=32, d_a=24, d_m=64)
        self.conv2 = GaANConv(in_channels=256, out_channels=dataset.num_classes,
                              heads=8, d_v=32, d_a=24, d_m=64)
        self.negative_slop = 0.1
        self.dropout = 0.1

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index), self.negative_slop)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index), self.negative_slop)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))
