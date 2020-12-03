import os.path as osp
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
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
parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
parser.add_argument('--attention_dropout', type=float, default=0.0005, help="dropout for gaan attention")
args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    
dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=args.attention_dropout, negative_slope=0.2)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=args.attention_dropout, negative_slope=0.2)

    def forward(self):
        x = F.dropout(data.x, p=args.dropout, training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


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
print(f"Final Test: {test_acc}")