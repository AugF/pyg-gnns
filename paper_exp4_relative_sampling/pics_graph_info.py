import os
import sys
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 8
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

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

dir_in = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp4_relative_sampling/batch_graph_info"
dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/new_exp_supplement"
file_out = "exp_sampling_minibatch_realtive_graph_info_"
# xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']
xticklabels = [1, 3, 6, 10, 25, 50]

for sampler in ["cluster", "graphsage"]:
    df_nodes, df_edges, df_degrees = {}, {}, {}
    for data in small_datasets:
        df_nodes[data], df_edges[data], df_degrees[data] = [], [], []
        if sampler == 'cluster':
            for cs in cluster_batchs:
                file_path = dir_in + "/" + sampler + "_gcn_" + data + "_" + str(cs) + ".log"
                print(file_path)
                if not os.path.exists(file_path):
                    df_nodes[data].append(np.nan)
                    df_edges[data].append(np.nan)
                    df_degrees[data].append(np.nan)
                    continue
                nodes, edges = 0, 0
                cnt = 0
                with open(file_path) as f:
                    for line in f:
                        match_line = re.match(r"nodes: (.*), edges: (.*)", line)
                        if match_line:
                            nodes += int(match_line.group(1))
                            edges += int(match_line.group(2))
                            cnt += 1
                nodes //= cnt
                edges //= cnt
                df_nodes[data].append(nodes)
                df_edges[data].append(edges)
                df_degrees[data].append(edges / nodes)
        else:
            for gs in graphsage_batchs[data]:
                file_path = dir_in + "/" + sampler + "_gcn_" + data + "_" + str(gs) + ".log"
                # print(file_path)
                if not os.path.exists(file_path):
                    df_nodes[data].append(np.nan)
                    df_edges[data].append(np.nan)
                    df_degrees[data].append(np.nan)
                    continue
                nodes, edges = 0, 0
                cnt = 0
                with open(file_path) as f:
                    for line in f:
                        match_line = re.match(r"nodes: (.*), edges: (.*)", line)
                        if match_line:
                            nodes += int(match_line.group(1))
                            edges += int(match_line.group(2))
                            cnt += 1
                nodes //= cnt
                edges //= cnt
                df_nodes[data].append(nodes)
                df_edges[data].append(edges)
                df_degrees[data].append(edges / nodes)
    
    # pics nodes
    fig, axes = plt.subplots(1, 3, figsize=(7, 6/3), tight_layout=True)
    ylabels = ["Number of Vertices", "Number of Edges", "Average Degree"]
    dfs = [df_nodes, df_edges, df_degrees]
    xlabel = "Relative Batch Size (%)"

    for k in range(3):
        ax = axes[k]
        df = dfs[k]
        markers = 'oD^sdp'
        colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, len(small_datasets)))
        for i, data in enumerate(small_datasets):
            ax.plot(xticklabels, df[data], 
                    color=colors[i], marker=markers[i], markersize=4, label=datasets_maps[data])
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabels[k], fontsize=10)
        ax.legend(fontsize="small")
        
    fig.savefig(dir_out + "/" + file_out + sampler + "_gcn.png")
    fig.savefig(dir_out + "/" + file_out + sampler + "_gcn.pdf")
        
                
        
