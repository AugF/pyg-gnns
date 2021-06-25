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
parser.add_argument('--infer_batch_size', type=int, default=1024, help='infer batch size')
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
    
print(f'begin loader, infer batch size: {args.infer_batch_size}')  

subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=args.infer_batch_size,
                                  shuffle=False, num_workers=args.num_workers)

print('begin model')
# 2. model
if args.model == 'gcn':
    norm = gcn_norm(data.edge_index, data.x.shape[0])
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

# not consider
@torch.no_grad()
def test(df=None):  # Inference should be performed on the full graph.
    model.eval()

    out = model.inference(data.x, subgraph_loader, df=df)
    # y_true = data.y.cpu()
    # y_pred = out.argmax(dim=-1)

    # accs = []
    # for mask in [data.train_mask, data.val_mask, data.test_mask]:
    #     correct = y_pred[mask].eq(y_true[mask]).sum().item()
    #     accs.append(correct / mask.sum().item())
    return


df = defaultdict(list)
loader_cnt = len(subgraph_loader)

print('begin eval')
for epoch in range(args.epochs):
    if loader_cnt * epoch >= 40: # 取40轮batch进行分析
        break
    test(df)

print(df)
pd.DataFrame(df).to_csv(args.real_path)
