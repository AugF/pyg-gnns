import sys
import time
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from torch_geometric.data import NeighborSampler
from utils import small_datasets, get_dataset, get_split_by_file

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp7_inference_sampling/exp_graphsage_hop"
file_out = "exp_inference_sampling_degrees_distribution_"
colors = 'rgb'
markers = 'oD^sdp'

def helper(edge_index, degrees):
    d_v = {}
    for e in edge_index.T.numpy():
        # 统计入度
        if e[0] not in d_v.keys():
            d_v[e[0]] = 1
        else:
            d_v[e[0]] += 1
        # 统计出度
        if e[1] not in d_v.keys():
            d_v[e[1]] = 1
        else:
            d_v[e[1]] += 1
    
    for k in d_v.keys():
        if d_v[k] not in degrees.keys():
            degrees[d_v[k]] = 1
        else:
            degrees[d_v[k]] += 1
    return degrees


def get_k_hop(k):
    for data_name in small_datasets:
        data = get_dataset(data_name, normalize_features=True)[0]

        # add train, val, test split
        if data_name in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
            file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', data_name + "/raw/role.json")
            data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)
        
        subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1] * 3, batch_size=1024,
                                        shuffle=False, num_workers=40)
        
        total_batches = len(subgraph_loader)
        total_degrees = {}
        for i in range(k):
            total_degrees[i] = {}
        
        cnt = 0
        for batch in subgraph_loader:
            if cnt == 5:
                batch_size, n_id, adjs = batch
                for i, (edge_index, e_id, size) in enumerate(adjs):
                    total_degrees[i] = helper(edge_index, total_degrees[i])
                break
            cnt += 1
        
        # 画图
        if not osp.exists(dir_out):
            os.makedirs(dir_out)
        
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(ymin=0.5, ymax=1e4)
        ax.set_xlim(xmin=0.5, xmax=1e4)
        ax.set_xlabel("Degrees")
        ax.set_ylabel("Number of Vertices")
        
        labels = [str(i) + '-hop' for i in range(k, 0, -1)] 
        # adjs[0]对应的是最远跳
        for i in range(k):
            ax.scatter(total_degrees[i].keys(), [total_degrees[i][k] for k in total_degrees[i]], color=colors[i], label=labels[i], marker=markers[i])
        
        ax.legend()
        fig.savefig(dir_out + "/" + file_out + data_name + ".png")
        fig.savefig(dir_out + "/" + file_out + data_name + ".pdf")


get_k_hop(3)