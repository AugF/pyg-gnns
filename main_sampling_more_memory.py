import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from collections import defaultdict
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

import sys
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import os.path as osp
import pandas as pd
from gaan.models import GaAN
from ggnn.models import GGNN
from gat.models import GAT
from gcn.models import GCN

from utils import get_dataset, gcn_norm, normalize, get_split_by_file, nvtx_push, nvtx_pop, log_memory, small_datasets
from utils import cross_entropy_loss, bce_with_logits_loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")

parser.add_argument('--model', type=str, default='gcn', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--epochs', type=int, default=2, help="epochs for training")
parser.add_argument('--layers', type=int, default=2, help="layers for hidden layer")
parser.add_argument('--hidden_dims', type=int, default=64, help="hidden layer output dims")
parser.add_argument('--heads', type=int, default=8, help="gat or gaan model: heads")
parser.add_argument('--head_dims', type=int, default=8, help="gat model: head dims") # head_dims * heads = hidden_dims
parser.add_argument('--d_v', type=int, default=8, help="gaan model: vertex's dim") # d_v * heads = hidden_dims?
parser.add_argument('--d_a', type=int, default=8, help="gaan model: each vertex's dim in edge attention") # d_a = head_dims
parser.add_argument('--d_m', type=int, default=64, help="gaan model: gate: max aggregator's dim, default=64")

parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--gpu', type=int, default=1, help='cpu:-1, cuda_id')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")
parser.add_argument('--json_path', type=str, default='', help="json file path for memory")
parser.add_argument('--mode', type=str, default='cluster', help='sampling: [cluster, graphsage]')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--relative_batch_size', type=float, default=None, help='number of cluster partitions per batch')
parser.add_argument('--batch_partitions', type=int, default=20, help='number of cluster partitions per batch')
parser.add_argument('--cluster_partitions', type=int, default=1500, help='number of cluster partitions')
parser.add_argument('--num_workers', type=int, default=40, help='number of Data Loader partitions')
parser.add_argument('--real_path', type=str, default='', help='memory path')
args = parser.parse_args()
gpu = args.gpu >= 0 and torch.cuda.is_available()
flag = not args.json_path == ''

print(args)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
device = torch.device(f'cuda: {args.gpu}' if gpu else 'cpu') # todo: model's device

# 0. set manual seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if gpu:
    torch.cuda.manual_seed(args.seed)

# 1. set datasets
dataset_info = args.dataset.split('_')
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    args.dataset = dataset_info[0]

dataset = get_dataset(args.dataset, normalize_features=True)
data = dataset[0]

# add train, val, test split
if args.dataset in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
    file_path = osp.join("/mnt/data/wangzhaokang/wangyunpan/data", args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

num_features = dataset.num_features
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    file_path = osp.join("/mnt/data/wangzhaokang/wangyunpan/data", "data/feats_x/" + '_'.join(dataset_info) + '_feats.npy')
    if osp.exists(file_path):
        data.x = torch.from_numpy(np.load(file_path)).to(torch.float) # 因为这里是随机生成的，不考虑normal features
        num_features = data.x.size(1)
    
if args.relative_batch_size is not None:
    args.batch_size = int(data.num_nodes * args.relative_batch_size + 0.5)
    args.batch_partitions = int(args.cluster_partitions * args.relative_batch_size)
    
print(f'begin loader, batch_size: {args.batch_size}, batch_partitions: {args.batch_partitions}')  

loader_time = time.time()    
if args.mode == 'cluster':
    # inductive
    row, col = [], []
    for i in range(data.edge_index.shape[1]):
        e = data.edge_index[:, i]
        if data.train_mask[e[0]] and data.train_mask[e[1]]:
            row.append(e[0])
            col.append(e[1])
    
    data.edge_index = torch.tensor([row, col])
    cluster_data = ClusterData(data, num_parts=args.cluster_partitions, recursive=False,
                            save_dir=dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=args.batch_partitions, shuffle=True,
                                num_workers=args.num_workers)
elif args.mode == 'graphsage':
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[25, 10], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers) # inductive learning
loader_time = time.time() - loader_time

print('begin model')
# 2. model
if args.model == 'gcn':
    if args.mode == 'graphsage':
        norm = gcn_norm(data.edge_index, data.x.shape[0])
    else:
        norm = None
    model = GCN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu, flag=flag, device=device, norm=norm,
        cluster_flag=args.mode == 'cluster'
    )
elif args.model == 'gat':
    model = GAT(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        head_dims=args.head_dims, heads=args.heads, gpu=gpu, flag=flag, device=device
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

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch, cnt, df=None):
    model.train()
    
    total_nodes = int(data.train_mask.sum())

    total_loss = 0

    train_iter = iter(train_loader)
    while True:
        try:
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]
            optimizer.zero_grad()
            batch = next(train_iter)
            
            if args.mode == "cluster":
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                y_pred, y_real = out[batch.train_mask], batch.y[batch.train_mask]
                if args.dataset in ['yelp']:
                    loss = bce_with_logits_loss(y_pred, y_real)
                else:
                    loss = cross_entropy_loss(y_pred, y_real)
                batch_size = batch.train_mask.sum().item()
            elif args.mode == 'graphsage':
                batch_size, n_id, adjs = batch
                adjs = [adj.to(device) for adj in adjs] # 这里等于成熟
                x = data.x[n_id].to(device)
                y = data.y[n_id[:batch_size]].to(device)
                optimizer.zero_grad()            
                out = model(data.x[n_id].to(device), adjs)
                y_pred, y_real = out, y
                if args.dataset in ['yelp']:
                    loss = bce_with_logits_loss(y_pred, y_real)
                else:
                    loss = cross_entropy_loss(y_pred, y_real)
            memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
            if df is not None:
                df['loss'].append(loss.item())
                df['memory'].append(memory)
                df['diff_memory'].append(memory - current_memory)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size    
            print(f'Batch: {cnt}, loss: {loss.item()}, cnt: {cnt}')                    
            cnt += 1  
            if cnt >= 40:
                break    
        except StopIteration:
            break

    loss = total_loss / total_nodes
    return loss, cnt



df = defaultdict(list)
cnt = 0
print('begin train')
for epoch in range(args.epochs):
    if cnt >= 40: # 取40轮batch进行分析
        break
    loss, cnt = train(epoch, cnt, df)

pd.DataFrame(df).to_csv(args.real_path)
