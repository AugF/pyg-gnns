import torch
import torch.nn.functional as F
import numpy as np
import torch.cuda.nvtx as nvtx
import argparse

from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN
from utils import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--random_splits', type=bool, default=False)

parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hidden_dims', type=int, default=128)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--d_v', type=int, default=24)
parser.add_argument('--d_a', type=int, default=24)
parser.add_argument('--d_m', type=int, default=64)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--negative_slop', type=float, default=0.1)

args = parser.parse_args()

args.gpu = args.gpu and torch.cuda.is_available()

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu:
    torch.cuda.manual_seed(args.seed)

# 1. load data
dataset = get_dataset(args.dataset, args.random_splits)
data = dataset[0]

# 2. model
if args.model == 'gcn':
    model = GCN(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, dropout=args.dropout
    )
elif args.model == 'gat':
    model = GAT(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims / args.heads, heads=args.heads,
        dropout=args.dropout, negative_slop=args.negative_slop
    )
elif args.model == 'ggnn':
    model = GGNN(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims
    )
elif args.model == 'gaan':
    model = GaAN(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims,
        heads=args.head, d_v=args.d_v,  # todo
        d_a=args.d_a, d_m=args.d_m
    )

# set to gpu
device = torch.device('cuda' if args.gpu else 'cpu')
model, data = model.to(device), data.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)


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

if not args.gpu:
    for epoch in range(10):
        train()
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *test()))
else:
    with torch.cuda.profiler.profile():
        train()
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            for epoch in range(10):
                nvtx.range_push("epoch " + epoch)
                nvtx.range_push("train")
                train()
                nvtx.range_pop()
                nvtx.range_push("eval")
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, *test()))
                nvtx.range_pop()
                nvtx.range_pop()





