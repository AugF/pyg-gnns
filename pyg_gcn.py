import os.path as osp
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from utils import get_dataset, get_split_by_file

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--epochs', type=int, default=1000, help="epochs for training")
parser.add_argument('--dataset', type=str, default="cora",
                    help='dataset')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.001, help="adam's weight decay")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    
dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

# if args.use_gdc:
#     gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
#                 normalization_out='col',
#                 diffusion_kwargs=dict(method='ppr', alpha=0.05),
#                 sparsification_kwargs=dict(method='topk', k=128,
#                                            dim=0), exact=True)
#     data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 64, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(64, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.convs[i].parameters(), weight_decay=args.weight_decay if i == 0 else 0)
    for i in range(2)]
, lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(args.epochs):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.8f}, Val: {:.8f}, Test: {:.8f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
print(f"Final Test: {test_acc:.8f}")