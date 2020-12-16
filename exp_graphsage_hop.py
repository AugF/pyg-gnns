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
colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, 5))
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
    
for data_name in small_datasets:
    data = get_dataset(data_name, normalize_features=True)[0]

    # add train, val, test split
    if data_name in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
        file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', data_name + "/raw/role.json")
        data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)
    
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1] * 5, batch_size=1024,
                                    shuffle=False, num_workers=40)
    
    total_batches = len(subgraph_loader)
    total_degrees = {}
    for i in range(5):
        total_degrees[i] = {}
        
    for batch in subgraph_loader:
        batch_size, n_id, adjs = batch
        for i, (edge_index, e_id, size) in enumerate(adjs):
            total_degrees[i] = helper(edge_index, total_degrees[i])
        break
    
    # for i in range(5):
    #     for k in total_degrees[i].keys():
    #         total_degrees[i][k] = int(total_degrees[i][k] / total_batches)
        
    # 画图
    if not osp.exists(dir_out + "/" + data_name):
        os.makedirs(dir_out + "/" + data_name)
    
    fig, ax = plt.subplots()
    ax.set_xscale("symlog", basex=10)
    ax.set_yscale("symlog", basey=10)
    ax.set_xlabel("Degrees")
    ax.set_ylabel("Number of Vertices")
    
    labels = ['5-hop', '4-hop', '3-hop', '2-hop', '1-hop']
    for i, c in enumerate(colors):
        ax.scatter(total_degrees[i].keys(), [total_degrees[i][k] for k in total_degrees[i]], color=c, label=labels[i], marker=markers[i])
    
    ax.legend()
    fig.savefig(dir_out + "/" + data_name + "/" + file_out + data_name + ".png")
