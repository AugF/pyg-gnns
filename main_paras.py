"""
2020-6-7版本的代码
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
from utils import get_dataset, get_split_by_file, nvtx_push, nvtx_pop, log_memory, small_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help="dataset: [cora, flickr, com-amazon, reddit, com-lj,"
                                                                    "amazon-computers, amazon-photo, coauthor-physics, pubmed]")

parser.add_argument('--model', type=str, default='gcn', help="gnn models: [gcn, ggnn, gat, gaan]")
parser.add_argument('--epochs', type=int, default=50, help="epochs for training")
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

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)

log_memory(flag, device, 'data load')

def train(epoch):
    t = time.time()

    # add forward start
    log_memory(flag, device, 'forward_start')

    nvtx_push(gpu, "forward")
    model.train()
    out = model(data.x, data.edge_index)
    nvtx_push(gpu, "loss")
    loss = F.nll_loss(F.log_softmax(out, dim=1)[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    nvtx_pop(gpu)
    nvtx_pop(gpu)
    log_memory(flag, device, 'forward_end')

    # add forward end
    nvtx_push(gpu, "backward")
    loss.backward()
    optimizer.step()
    nvtx_pop(gpu)

    # add backward end
    log_memory(flag, device, 'backward_end')

    log = 'Epoch: {:03d}, train_loss: {:.8f}, train_time: {:.4f}s'
    t = time.time() - t
    print(log.format(epoch, loss.data.item(), t))
    return 

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    log_memory(flag, device, 'other_start')    
    nvtx_push(gpu, "other")
    logits, accs = F.log_softmax(out, dim=1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    nvtx_pop(gpu)
    return accs

if not gpu:
    for epoch in range(args.epochs + 1):
        train(epoch)
        log = 'Accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(*test()))
else:
    with torch.cuda.profiler.profile():
        train(-1)
        log_memory(flag, device, 'eval_end')
        with torch.autograd.profiler.emit_nvtx(record_shapes=not args.no_record_shapes):
            for epoch in range(args.epochs):
                nvtx_push(gpu, "epochs" + str(epoch))
                nvtx_push(gpu, "train")
                train(epoch)
                nvtx_pop(gpu)
                nvtx_push(gpu, "eval")
                log = 'Accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(*test()))
                
                # add 
                log_memory(flag, device, 'eval_end')
                
                nvtx_pop(gpu)
                nvtx_pop(gpu)
                
    if flag:
        from utils import df
        with open(args.json_path, "w") as f:
            json.dump(df, f)