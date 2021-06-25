import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
import json
import torch_geometric.transforms as T

from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

from datasets import DataProcess
from citation_datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.coauthor import Coauthor

import torch.cuda.nvtx as nvtx

memory_labels = ["allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current", "reserved_bytes.all.peak"]
small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
small_nodes = [7650, 19717, 13752, 34493, 89250, 334863]

cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
}

datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

df = {}

def get_dataset(name, normalize_features=False, transform=None): #
    if name in ["cora", "pubmed"]:
        path = osp.join('/mnt/data/wangzhaokang/wangyunpan/data')
        dataset = Planetoid(path, name, split='full')
    else:
        path = osp.join('/mnt/data/wangzhaokang/wangyunpan/data', name)
        if name in ["amazon-computers", "amazon-photo"]:
            dataset = Amazon(path, name[7:])
        elif name == "coauthor-physics":
            dataset = Coauthor(path, name[9:])
        else:  # [com-amazon, com-lj, flickr, reddit, yelp]
            dataset = DataProcess(root=path)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_train_val_test_split(nodes, tr, va, seed=1):
    np.random.seed(seed)
    idx = np.arange(nodes)
    np.random.shuffle(idx)
    tr, va = int(nodes * tr), int(nodes * (tr + va))

    train_mask = torch.zeros(nodes, dtype=torch.bool)
    train_mask[torch.tensor(idx[: tr])] = True

    val_mask = torch.zeros(nodes, dtype=torch.bool)
    val_mask[torch.tensor(idx[tr: va])] = True

    test_mask = torch.zeros(nodes, dtype=torch.bool)
    test_mask[torch.tensor(idx[va:])] = True
    return train_mask, val_mask, test_mask


def get_split_by_file(file_path, nodes): # 通过读取roles.json文件来获取train, val, test mask
    with open(file_path) as f:
        role = json.load(f)

    train_mask = torch.zeros(nodes, dtype=torch.bool)
    train_mask[torch.tensor(role['tr'])] = True

    val_mask = torch.zeros(nodes, dtype=torch.bool)
    val_mask[torch.tensor(role['va'])] = True

    test_mask = torch.zeros(nodes, dtype=torch.bool)
    test_mask[torch.tensor(role['te'])] = True
    return train_mask, val_mask, test_mask


def gcn_norm(edge_index, num_nodes, edge_weight=None, improved=False,
            dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes) # ? todo 

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def gcn_cluster_norm(edge_index, num_nodes, edge_weight=None, improved=False,
            dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    #fill_value = 1 if not improved else 2
    #edge_index, edge_weight = add_remaining_self_loops(
    #    edge_index, edge_weight, fill_value, num_nodes) # ? todo 

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
def cal_sparse(x, m, n):
    x_bool = np.where(x == 0, 1, 0)
    return x_bool.mean(), m / n


def nvtx_push(flag, info):
    if flag:
        nvtx.range_push(info)


def nvtx_pop(flag):
    if flag:
        nvtx.range_pop()


def log_memory(flag, device, label):
    if flag:
        res = torch.cuda.memory_stats(device)
        torch.cuda.reset_max_memory_allocated(device)
        # print(res["allocated_bytes.all.current"])
        if label not in df.keys():
            df[label] = [[res[i] for i in memory_labels]]
        else:
            df[label].append([res[i] for i in memory_labels])


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def cross_entropy_loss(y_pred, y_true):
    y_true = y_true.long()
    y_pred = torch.nn.functional.log_softmax(y_pred, dim=-1)
    return torch.nn.functional.nll_loss(y_pred, y_true)


def bce_with_logits_loss(y_pred, y_true, reduction="mean"):
    y_true = y_true.float()
    loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)(y_pred, y_true)
    if reduction == "none":
        loss = torch.sum(torch.mean(loss, dim=0))
    return loss      


if __name__ == '__main__':
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    for data in datasets:
        dataset = get_dataset(data, normalize_features=False)[0]
        print(data, cal_sparse(dataset.x, dataset.num_edges, dataset.num_nodes))

