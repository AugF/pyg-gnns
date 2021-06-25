import sys
import time
import os
import numpy as np
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.data import NeighborSampler
from utils import get_dataset, get_split_by_file, small_datasets, datasets_maps

plt.style.use("ggplot")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

dir_out = "paper_exp7_inference_sampling"
datasets = ["amazon-photo", "coauthor-physics", "com-amazon"]
xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL GRAPH']
cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
}


def save_to_csv():
    df_nodes, df_edges, df_degrees = {}, {}, {}
    # for data_name in datasets:
    for data_name in small_datasets:
        print(data_name)
        dataset = get_dataset(data_name, normalize_features=True)
        data = dataset[0] # 备注: 这里得出的结果是无向图，即如果是有向图，会被处理为无向图
        
        nodes, edges = data.num_nodes, data.num_edges
        if data_name in ['amazon-computers', 'amazon-photo', 'coauthor-physics']:
            file_path = osp.join('/home/wangzhaokang/wangyunpan/gnns-project/datasets', data_name + "/raw/role.json")
            data.train_mask, data.val_mask, data.test_mask = get_split_by_file(file_path, data.num_nodes)

        df_nodes[f'{data_name}_neighbor_sampler'] = []
        df_nodes[f'{data_name}_inference_sampler'] = [] 
        df_edges[f'{data_name}_neighbor_sampler'] = []
        df_edges[f'{data_name}_inference_sampler'] = [] 
        df_degrees[f'{data_name}_neighbor_sampler'] = []
        df_degrees[f'{data_name}_inference_sampler'] = [] 
        
        for gs in graphsage_batchs[data_name]:
            # train_loader
            train_loader = NeighborSampler(data.edge_index, node_idx=None,
                                    sizes=[25, 10], batch_size=gs, shuffle=True) 
            
            # subgraph_loader
            subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1] * 2, batch_size=gs,
                                    shuffle=False)
            
            flag = False
            train_nodes, train_edges = 0, 0
            for runs in range(50):
                for batch in train_loader:
                    batch_size, n_id, adjs = batch
                    train_nodes += adjs[0][2][0]
                    train_edges += adjs[0][0].shape[1]
                    break
            train_nodes //= 50
            train_edges //= 50

            flag = False
            subgraph_cnt = 0
            subgraph_nodes, subgraph_edges = 0, 0
            for runs in range(50):
                for batch in subgraph_loader:
                    batch_size, n_id, adjs = batch
                    subgraph_nodes += adjs[0][2][0]
                    subgraph_edges += adjs[0][0].shape[1]
                    subgraph_cnt += 1
                    break
            subgraph_nodes //= 50
            subgraph_edges //= 50
            
            df_nodes[f'{data_name}_neighbor_sampler'].append(train_nodes)
            df_nodes[f'{data_name}_inference_sampler'].append(subgraph_nodes) 
            df_edges[f'{data_name}_neighbor_sampler'].append(train_edges)
            df_edges[f'{data_name}_inference_sampler'].append(subgraph_edges) 
            df_degrees[f'{data_name}_neighbor_sampler'].append(train_edges / train_nodes)
            df_degrees[f'{data_name}_inference_sampler'].append(subgraph_edges / subgraph_nodes) 
        
        df_nodes[f'{data_name}_neighbor_sampler'].append(nodes)
        df_nodes[f'{data_name}_inference_sampler'].append(nodes) 
        df_edges[f'{data_name}_neighbor_sampler'].append(edges)
        df_edges[f'{data_name}_inference_sampler'].append(edges) 
        df_degrees[f'{data_name}_neighbor_sampler'].append(edges / nodes)
        df_degrees[f'{data_name}_inference_sampler'].append(edges / nodes)
    pd.DataFrame(df_nodes, index=xticklabels).to_csv(dir_out + "/inference_graph_info_vertices.csv")
    pd.DataFrame(df_edges, index=xticklabels).to_csv(dir_out + "/inference_graph_info_edges.csv")
    pd.DataFrame(df_degrees, index=xticklabels).to_csv(dir_out + "/inference_graph_info_avg_degrees.csv")
 
 
def pics_inference_graph_info(file_name, ylabel, file_class, ymax, dir_save="paper_exp7_inference_sampling"):
    print(file_name)
    df = pd.read_csv(dir_out + "/" + file_name, index_col=0)
    
    # neighbor sampler
    # fig, ax = plt.subplots(tight_layout=True)
    # ax.set_ylabel(ylabel)
    # ax.set_xlabel("Relative Batch Size (%)")
    markers = 'oD^sdp'
    # ax.set_ylim(0, ymax)
    colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, len(small_datasets)))
    # for i, data in enumerate(small_datasets):
    #     ax.plot(xticklabels, df[f'{data}_neighbor_sampler'], 
    #             color=colors[i], marker=markers[i], label=datasets_maps[data])

    # ax.set_xticklabels(xticklabels)
    # ax.legend(fontsize='small')
    # fig.savefig(dir_save + "/exp_inference_sampling_graph_info_neighbor_sampler_" + file_class + ".png")
    # fig.savefig(dir_save + "/exp_inference_sampling_graph_info_neighbor_sampler_" + file_class + ".pdf")
    
    # inference sampler
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Relative Batch Size (%)", fontsize=16)
    # ax.set_ylim(0, ymax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for i, data in enumerate(small_datasets):
        ax.plot(xticklabels, df[f'{data}_inference_sampler'], 
                color=colors[i], marker=markers[i], label=datasets_maps[data])

    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.legend(fontsize=10)
    fig.savefig(dir_save + "/exp_inference_sampling_graph_info_inference_sampler_" + file_class + ".png")
    fig.savefig(dir_save + "/exp_inference_sampling_graph_info_inference_sampler_" + file_class + ".pdf")

def get_avg_degrees_csv():
    df_nodes = pd.read_csv(dir_out + "/inference_graph_info_vertices.csv", index_col=0)
    df_edges = pd.read_csv(dir_out + "/inference_graph_info_edges.csv", index_col=0)
    
    df_degrees = {}
    for c in df_nodes.columns:
        df_degrees[c] = [df_edges[c][i] / df_nodes[c][i] for i in range(len(df_nodes.index))]
    pd.DataFrame(df_degrees, index=xticklabels).to_csv(dir_out + "/inference_graph_info_avg_degrees.csv")

if __name__ == "__main__":
    # save_to_csv()
    # pics_inference_graph_info("inference_graph_info_vertices.csv", "Number of Vertices", "vertices", 350000)
    pics_inference_graph_info("inference_graph_info_edges.csv", "Number of Edges", "edges", 980000, dir_save="/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/exp_supplement")
    # get_avg_degrees_csv()
    pics_inference_graph_info("inference_graph_info_avg_degrees.csv", "Average Degree", "avg_degrees", 37, dir_save="/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/exp_supplement")