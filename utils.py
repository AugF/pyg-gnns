import os.path as osp

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from datasets import DataProcess

import torch.cuda.nvtx as nvtx


def get_dataset(name, normalize_features=False, transform=None): #
    if name == "cora":
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = Planetoid(path, name)
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
        dataset = DataProcess(root=path)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


def nvtx_push(flag, info):
    if flag:
        nvtx.range_push(info)

def nvtx_pop(flag):
    if flag:
        nvtx.range_pop()


if __name__ == '__main__':
    data = get_dataset("reddit")