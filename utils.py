import os.path as osp

import numpy as np
import torch
import json
import torch_geometric.transforms as T
from datasets import DataProcess
from citation_datasets import Planetoid
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.coauthor import Coauthor

import torch.cuda.nvtx as nvtx


def get_dataset(name, normalize_features=False, transform=None): #
    if name in ["cora", "pubmed"]:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = Planetoid(path, name, split='full')
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
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

def nvtx_push(flag, info):
    if flag:
        nvtx.range_push(info)


def nvtx_pop(flag):
    if flag:
        nvtx.range_pop()


if __name__ == '__main__':
    data = get_dataset("reddit")

