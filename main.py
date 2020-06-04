import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time

from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN
from utils import get_dataset, get_split_by_file, nvtx_push, nvtx_pop
from preprocess import get_role

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")

parser.add_argument('--model', type=str, default='gat', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--epochs', type=int, default=20, help="epochs for training")
parser.add_argument('--layers', type=int, default=4, help="layers for hidden layer")
parser.add_argument('--hidden_dims', type=int, default=64, help="hidden layer output dims")
parser.add_argument('--heads', type=int, default=8, help="gat or gaan model: heads")
parser.add_argument('--head_dims', type=int, default=8, help="gat model: head dims") # head_dims * heads = hidden_dims
parser.add_argument('--d_v', type=int, default=8, help="gaan model: vertex's dim") # d_v * heads = hidden_dims?
parser.add_argument('--d_a', type=int, default=8, help="gaan model: each vertex's dim in edge attention") # d_a = head_dims
parser.add_argument('--d_m', type=int, default=64, help="gaan model: gate: max aggregator's dim, default=64")

parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu, not use gpu')
parser.add_argument('--device', type=str, default='cuda:0', help='[cpu, cuda:id]')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")

args = parser.parse_args()
gpu = not args.cpu and torch.cuda.is_available()
print(args)

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu:
    torch.cuda.manual_seed(args.seed)

# 1. load epochs
dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

print(data.num_nodes, data.num_edges, dataset.num_features, dataset.num_classes)
# add train, val, test split
if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
    raw_dir = "data/" + args.dataset + '/raw'
    get_role(raw_dir=raw_dir, nodes=data.num_nodes, tr=0.50, va=0.25)
    file_path = "data/" + args.dataset + "/raw/role.json"
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

# 2. model
if args.model == 'gcn':
    model = GCN(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu
    )
elif args.model == 'gat':
    model = GAT(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        head_dims=args.head_dims, heads=args.heads, gpu=gpu
    )
elif args.model == 'ggnn':
    model = GGNN(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu
    )
elif args.model == 'gaan':
    model = GaAN(
        layers=args.layers,
        n_features=dataset.num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims,
        heads=args.heads, d_v=args.d_v,
        d_a=args.d_a, d_m=args.d_m, gpu=gpu
    )

print(model)
# set to gpu
device = torch.device('cuda:0' if gpu else 'cpu')
model, data = model.to(device), data.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    nvtx_push(gpu, "forward")
    model.train()
    loss = F.nll_loss(F.log_softmax(model(data.x, data.edge_index), dim=1)[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    nvtx_pop(gpu)
    nvtx_push(gpu, "backward")
    loss.backward()
    optimizer.step()
    nvtx_pop(gpu)
    log = 'Epoch: {:03d}, train_loss: {:.8f}, train_time: {:.4f}s'
    t = time.time() - t
    print(log.format(epoch, loss.data.item(), t))
    return t


def test():
    model.eval()
    logits, accs = F.log_softmax(model(data.x, data.edge_index), dim=1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if not gpu:
    for epoch in range(args.epochs + 1):
        train(epoch)
        log = 'Accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(*test()))
else:
    with torch.cuda.profiler.profile():
        train(-1)
        with torch.autograd.profiler.emit_nvtx(record_shapes=not args.no_record_shapes):
            t = 0
            for epoch in range(args.epochs):
                nvtx_push(gpu, "epochs" + str(epoch))
                nvtx_push(gpu, "train")
                t += train(epoch)
                nvtx_pop(gpu)

                nvtx_push(gpu, "eval")
                log = 'Accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(*test()))
                nvtx_pop(gpu)
                nvtx_pop(gpu)
            print("Average train time: {}s".format(t / args.epochs))





