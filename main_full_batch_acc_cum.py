"""
跟main.py相比为full文件的对比版本
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import sys
import json
import os.path as osp

from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN
from sklearn.metrics import f1_score
from utils import get_dataset, get_split_by_file, nvtx_push, nvtx_pop, log_memory, small_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")

parser.add_argument('--model', type=str, default='gcn', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--runs', type=int, default=10, help="total runs")
parser.add_argument('--epochs', type=int, default=100000, help="epochs for training")
parser.add_argument('--layers', type=int, default=2, help="layers for hidden layer")
parser.add_argument('--hidden_dims', type=int, default=64, help="hidden layer output dims")
parser.add_argument('--heads', type=int, default=8, help="gat or gaan model: heads")
parser.add_argument('--head_dims', type=int, default=8, help="gat model: head dims") # head_dims * heads = hidden_dims
parser.add_argument('--d_v', type=int, default=8, help="gaan model: vertex's dim") # d_v * heads = hidden_dims?
parser.add_argument('--d_a', type=int, default=8, help="gaan model: each vertex's dim in edge attention") # d_a = head_dims
parser.add_argument('--d_m', type=int, default=64, help="gaan model: gate: max aggregator's dim, default=64")

parser.add_argument('--x_sparse', action='store_true', default=False, help="whether to use data.x sparse version")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu, not use gpu')
parser.add_argument('--device', type=str, default='cuda:0', help='[cpu, cuda:id]')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")
parser.add_argument('--json_path', type=str, default='', help="json file path for memory")
parser.add_argument('--fix_time', type=int, default=800, help="fix_time")


args = parser.parse_args()
gpu = not args.cpu and torch.cuda.is_available()
flag = not args.json_path == ''

print(args)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu:
    torch.cuda.manual_seed(args.seed)

# for feats experiment
dataset_info = args.dataset.split('_')
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    args.dataset = dataset_info[0]

# 1. load epochs
dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

# add train, val, test split
if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']: 
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)


# change transductive to inductive
# row, col = [], []
# for i in range(data.edge_index.shape[1]):
#     e = data.edge_index[:, i]
#     if data.train_mask[e[0]] and data.train_mask[e[1]]:
#         row.append(e[0])
#         col.append(e[1])
# data.edge_index = torch.tensor([row, col])

num_features = dataset.num_features
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', "data/feats_x/" + '_'.join(dataset_info) + '_feats.npy')
    if osp.exists(file_path):
        data.x = torch.from_numpy(np.load(file_path)).to(torch.float) # 因为这里是随机生成的，不考虑normal features
        num_features = data.x.size(1)

if args.x_sparse:
    data.x = data.x.to_sparse()

device = torch.device(args.device if gpu else 'cpu')

# 2. model
if args.model == 'gcn':
    model = GCN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu, flag=flag, device=device
    )
elif args.model == 'gat':
    model = GAT(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        head_dims=args.head_dims, heads=args.heads, gpu=gpu, flag=flag, sparse_flag=args.x_sparse, device=device
    )
elif args.model == 'ggnn':
    model = GGNN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu, flag=flag, device=device
    )
elif args.model == 'gaan':
    model = GaAN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims,
        heads=args.heads, d_v=args.d_v,
        d_a=args.d_a, d_m=args.d_m, gpu=gpu, flag=flag, device=device
    )

print(model)
# set to gpu
model, data = model.to(device), data.to(device)

def train(epoch):
    model.train()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(F.log_softmax(out, dim=1)[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    
    logits, accs = F.log_softmax(out, dim=1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam([
        dict(params=model.convs[i].parameters(), weight_decay=args.weight_decay if i == 0 else 0)
        for i in range(1 if args.model == "ggnn" else args.layers)]
        , lr=args.lr)  # Only perform weight-decay on first convolution, 参考了pytorch_geometric中的gcn.py的例子: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

    es_count = best_val_acc = test_acc = 0
    t0 = time.time()
    patient_step = 0
    for epoch in range(args.epochs):
        loss = train(epoch)
        accs = test()
        if accs[1] > best_val_acc:
            best_val_acc = accs[1]
            test_acc = accs[2]
        cur_time = time.time() - t0
        print(f"Batch: {epoch:03d}, loss:{loss:.8f}, train_acc: {accs[0]:.8f}, val_acc: {accs[1]:.8f}, best_val_acc: {best_val_acc: .8f}, best_test_acc: {test_acc:.8f}, cur_use_time: {cur_time:.4f}s")
        if cur_time > args.fix_time:
            sys.exit(0)
  