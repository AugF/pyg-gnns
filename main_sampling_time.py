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

from utils import get_dataset, gcn_norm, normalize, get_split_by_file, nvtx_push, nvtx_pop, log_memory, small_datasets

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
parser.add_argument('--gpu', type=int, default=0, help='cpu:-1, cuda_id')
parser.add_argument('--lr', type=float, default=0.01, help="adam's learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0005, help="adam's weight decay")
parser.add_argument('--no_record_shapes', action='store_false', default=True, help="nvtx or autograd's profile to record shape")
parser.add_argument('--json_path', type=str, default='', help="json file path for memory")
parser.add_argument('--mode', type=str, default='cluster', help='sampling: [cluster, graphsage]')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--batch_partitions', type=int, default=20, help='number of cluster partitions per batch')
parser.add_argument('--cluster_partitions', type=int, default=1500, help='number of cluster partitions')
parser.add_argument('--num_workers', type=int, default=40, help='number of Data Loader partitions')
args = parser.parse_args()
gpu = args.gpu >= 0 and torch.cuda.is_available()
flag = not args.json_path == ''

print(args)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

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
    file_path = osp.join("/data/wangzhaokang/wangyunpan", "data/" + args.dataset + "/raw/role.json")
    data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

num_features = dataset.num_features
if dataset_info[0] in small_datasets and len(dataset_info) > 1:
    file_path = osp.join("/data/wangzhaokang/wangyunpan", "data/feats_x/" + '_'.join(dataset_info) + '_feats.npy')
    if osp.exists(file_path):
        data.x = torch.from_numpy(np.load(file_path)).to(torch.float) # 因为这里是随机生成的，不考虑normal features
        num_features = data.x.size(1)
    
# 2. set sampling


# 2.2 train_data
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

# 3. set model
if args.model == 'gcn':
    # 预先计算edge_weight出来
    if args.mode == 'graphsage':
        norm = gcn_norm(data.edge_index, data.x.shape[0])
    else:
        norm = None
    model = GCN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu, flag=flag, norm=norm, cluster_flag=args.mode == 'cluster'
    )
elif args.model == 'gat':
    model = GAT(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        head_dims=args.head_dims, heads=args.heads, gpu=gpu, flag=flag
    )
elif args.model == 'ggnn':
    model = GGNN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims, gpu=gpu, flag=flag
    )
elif args.model == 'gaan':
    model = GaAN(
        layers=args.layers,
        n_features=num_features, n_classes=dataset.num_classes,
        hidden_dims=args.hidden_dims,
        heads=args.heads, d_v=args.d_v,
        d_a=args.d_a, d_m=args.d_m, gpu=gpu, flag=flag
    )

device = torch.device(f'cuda: {args.gpu}' if gpu else 'cpu') # todo: model's device
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    
    total_nodes = int(data.train_mask.sum())

    total_loss = 0

    sampling_time, to_time, train_time = 0.0, 0.0, 0.0

    train_iter = iter(train_loader)
    cnt = 0
    while True:
        try:
            t0 = time.time()
            batch = next(train_iter)
            t1 = time.time()
            sampling_time += t1 - t0
            nvtx_push(gpu, "batch" + str(cnt))
            log_memory(flag, device, 'forward_start')
            nvtx_push(gpu, "forward")
            
            if args.mode == "cluster":
                batch = batch.to(device)
                t2 = time.time()
                to_time += t2 - t1
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                if args.dataset in ['yelp', 'amazon']:
                    loss = torch.nn.BCEWithLogitsLoss()(out[batch.train_mask, :], batch.y[batch.train_mask, :])
                else:
                    loss = F.nll_loss(out.log_softmax(dim=-1)[batch.train_mask], batch.y[batch.train_mask])
                batch_size = batch.train_mask.sum().item()
            elif args.mode == 'graphsage':
                batch_size, n_id, adjs = batch
                adjs = [adj.to(device) for adj in adjs] # 这里等于成熟
                x = data.x[n_id].to(device)
                if args.dataset in ['yelp', 'amazon']:
                    y = data.y[n_id[:batch_size], :].to(device)
                else:
                    y = data.y[n_id[:batch_size]].to(device)
                t2 = time.time()
                to_time += t2 - t1
                optimizer.zero_grad()            
                out = model(data.x[n_id].to(device), adjs)
                if args.dataset in ['yelp', 'amazon']:
                    loss = torch.nn.BCEWithLogitsLoss()(out, y)
                else:
                    loss = F.nll_loss(out.log_softmax(dim=-1), y)
            nvtx_pop(gpu)
            
            log_memory(flag, device, 'forward_end')
            nvtx_push(gpu, "backward")
            loss.backward()
            optimizer.step()
            train_time += time.time() - t2
            nvtx_pop(gpu)
            
            log_memory(flag, device, 'backward_end')
            total_loss += loss.item() * batch_size 
            
            print(f"batch: ")           
            nvtx_pop(gpu)
            
            cnt += 1      
        except StopIteration:
            break

    #print("batchs", cnt, args.batch_size * cnt)
    #print("sampling", all_nodes, all_edges)
    #print("real", data.x.shape[0], data.edge_index.shape[1])
    loss = total_loss / total_nodes
    return loss, sampling_time, to_time, train_time, cnt

# not consider
@torch.no_grad()
def test():  # Inference should be performed on the full graph.
    model.eval()

    out = model.inference(data.x, subgraph_loader)
    y_true = y.cpu()
    y_pred = out.argmax(dim=-1)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(y_true[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs


if gpu:
    with torch.cuda.profiler.profile():
        train(-1)
        log_memory(flag, device, 'warmup end')
        with torch.autograd.profiler.emit_nvtx(record_shapes=not args.no_record_shapes):
            avg_batch_sampling_time = 0
            avg_batch_to_time = 0
            avg_batch_train_time = 0
            count = 0
            for epoch in range(args.epochs):
                if count >= 50: # 取50轮batch进行分析
                   break
                nvtx_push(gpu, "epochs" + str(epoch))
                nvtx_push(gpu, "train")
                loss, sampling_time, to_time, train_time, cnt = train(epoch)
                #loss, batch_times, cnt = train(epoch)
                count += cnt
                
                nvtx_pop(gpu)
                
                print(f"loss: {loss}" )
                
                #nvtx_push(gpu, "eval")
                #train_acc, val_acc, test_acc = test()
                #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                #         f'Val: {val_acc:.4f}, test: {test_acc:.4f}')
                # # add 
                #nvtx_pop(gpu)
                nvtx_pop(gpu)
                avg_batch_sampling_time += sampling_time
                avg_batch_to_time += to_time
                avg_batch_train_time += train_time

            avg_batch_sampling_time /= count
            avg_batch_to_time /= count
            avg_batch_train_time /= count
            
            print(f"loader_time:{loader_time}, avg_batch_train_time: {avg_batch_train_time}, avg_batch_sampling_time:{avg_batch_sampling_time}, avg_batch_to_time: {avg_batch_to_time}")
            #print(f"loader_time:{loader_time}, avg_epoch_train_time: {avg_epoch_train_time}, avg_epochs_sampling_time:{avg_epoch_sampling_time}, avg_epoch_to_time: {avg_epoch_to_time}")
    if flag:
        from utils import df
        with open(args.json_path, "w") as f:
            json.dump(df, f)
