"""
Sampler背景下，精度与BatchSize变化的文件, fix_time
"""
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

import sys
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import os.path as osp
from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN
from logger import Logger

from utils import get_dataset, gcn_norm, normalize, get_split_by_file, nvtx_push, nvtx_pop, log_memory, small_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")

parser.add_argument('--model', type=str, default='gcn', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--runs', type=int, default=1, help="total runs")
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
parser.add_argument('--device', type=str, default='cuda:0', help='[cpu, cuda:id]')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu, not use gpu')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")
parser.add_argument('--json_path', type=str, default='', help="json file path for memory")
parser.add_argument('--mode', type=str, default='cluster', help='sampling: [cluster, graphsage]')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--batch_partitions', type=int, default=20, help='number of cluster partitions per batch')
parser.add_argument('--cluster_partitions', type=int, default=1500, help='number of cluster partitions')
parser.add_argument('--num_workers', type=int, default=40, help='number of Data Loader partitions')
parser.add_argument('--fix_time', type=int, default=800, help='fix use time')
args = parser.parse_args()
gpu = not args.cpu and torch.cuda.is_available()
flag = not args.json_path == ''

print(args)

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu:
    torch.cuda.manual_seed(args.seed)

device = torch.device(args.device if gpu else 'cpu')

# 1. set datasets
dataset_info = args.dataset.split('_')
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    args.dataset = dataset_info[0]

dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

# add train, val, test split
if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

num_features = dataset.num_features
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', "data/feats_x/" + '_'.join(dataset_info) + '_feats.npy')
    if osp.exists(file_path):
        data.x = torch.from_numpy(np.load(file_path)).to(torch.float) # 因为这里是随机生成的，不考虑normal features
        num_features = data.x.size(1)
    
# 2. set sampling
# 2.1 test_data
subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                  shuffle=False, num_workers=args.num_workers)

# 2.2 train_data: 为了统一对比，这里的结果为Transductive
if args.mode == 'cluster':
    cluster_data = ClusterData(data, num_parts=args.cluster_partitions, recursive=False,
                            save_dir=dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_partitions, shuffle=True,
                                num_workers=args.num_workers)
elif args.mode == 'graphsage':
    train_loader = NeighborSampler(data.edge_index, node_idx=None,
                               sizes=[25, 10], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)

# 3. set model
if args.model == 'gcn':
    # 预先计算edge_weight出来
    norm = gcn_norm(data.edge_index, data.x.shape[0])
    model = GCN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu, flag=flag, 
        device=device, cached_flag=False, norm=norm
    )
elif args.model == 'gat':
    model = GAT(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        head_dims=args.head_dims, heads=args.heads, gpu=gpu, flag=flag, sparse_flag=args.x_sparse, device=device,
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

model, data = model.to(device), data.to(device)

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

@torch.no_grad()
def test_sampling():  # Inference should be performed on the full graph.
    t0 = time.time()
    model.eval()
    out = model.inference(data.x, subgraph_loader)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1)
    t1 = time.time()
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(y_true[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

def train(optimizer, t0, bs_count, best_val_acc, test_acc):
    model.train()
    
    train_iter = iter(train_loader)
    while True:
        try:
            t1 = time.time()
            batch = next(train_iter)
            if args.mode == "cluster":
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out.log_softmax(dim=-1)[batch.train_mask], batch.y[batch.train_mask])
                batch_size = batch.train_mask.sum().item()
            elif args.mode == 'graphsage':
                batch_size, n_id, adjs = batch
                adjs = [adj.to(device) for adj in adjs] # 这里等于成熟
                x = data.x[n_id].to(device)
                y = data.y[n_id[:batch_size]].to(device)
                optimizer.zero_grad()            
                out = model(data.x[n_id].to(device), adjs)
                loss = F.nll_loss(out.log_softmax(dim=-1), y)
            
            loss.backward()
            optimizer.step()
            # 指定batch size下，进行汇报时间和精度
            bs_count += 1
            if args.dataset == "coauthor-physics":
                accs = test_sampling()
            else:
                accs = test()
            if accs[1] >= best_val_acc:
                best_val_acc = accs[1]
                test_acc = accs[2]
            cur_use_time = time.time() - t0
            print(f"Batch: {bs_count:03d}, loss:{loss.item():.8f}, train_acc: {accs[0]:.8f}, val_acc: {accs[1]:.8f}, best_val_acc: {best_val_acc: .8f}, best_test_acc: {test_acc:.8f}, cur_use_time: {cur_use_time:.4f}s")
        
            if cur_use_time > args.fix_time:
                sys.exit(0)   
        except StopIteration:
            break
    return bs_count, best_val_acc, test_acc


for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam([
        dict(params=model.convs[i].parameters(), weight_decay=args.weight_decay if i == 0 else 0)
        for i in range(1 if args.model == "ggnn" else args.layers)]
        , lr=args.lr)  # Only perform weight-decay on first convolution, 参考了pytorch_geometric中的gcn.py的例子: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py
    bs_count = 0
    best_val_acc = test_acc = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        bs_count, best_val_acc, test_acc = train(optimizer, t0, bs_count, best_val_acc, test_acc)
                    