import torch
import torch.nn.functional as F
import numpy as np
import argparse

from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN
from utils import get_dataset, nvtx_push, nvtx_pop

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')

parser.add_argument('--model', type=str, default='gaan')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hidden_dims', type=int, default=128)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--d_v', type=int, default=24) # d_v * heads = hidden_dims?
parser.add_argument('--d_a', type=int, default=24)
parser.add_argument('--d_m', type=int, default=64)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--negative_slop', type=float, default=0.1)

parser.add_argument('--record_shapes', type=bool, default=True)

args = parser.parse_args()

args.gpu = args.gpu and torch.cuda.is_available()

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.gpu:
    torch.cuda.manual_seed(args.seed)

# 1. load data
dataset = get_dataset(args.dataset)
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
        hidden_dims=args.hidden_dims // args.heads, heads=args.heads,
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
        heads=args.heads, d_v=args.d_v,  # todo
        d_a=args.d_a, d_m=args.d_m
    )

# set to gpu
device = torch.device('cuda' if args.gpu else 'cpu')
model, data = model.to(device), data.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)


def train():
    nvtx_push(args.gpu, "forward")
    model.train()
    loss = F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    nvtx_pop(args.gpu)
    nvtx_push(args.gpu, "backward")
    loss.backward()
    optimizer.step()
    nvtx_pop(args.gpu)


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
        with torch.autograd.profiler.emit_nvtx(record_shapes=args.record_shapes):
            for epoch in range(10):
                nvtx_push(args.gpu, "epochs" + str(epoch))
                nvtx_push(args.gpu, "train")
                train()
                nvtx_pop(args.gpu)
                nvtx_push(args.gpu, "eval")
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, *test()))
                nvtx_pop(args.gpu)
                nvtx_pop(args.gpu)





