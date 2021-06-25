import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.patches import Polygon, Patch

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
re_percents = [1024, 2048, 4096, 8192]
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "batch_more_inference_memory_figs")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'reddit', 'yelp']
color = dict(medians='red', boxes='blue', caps='black')

batch_sizes = [1024, 2048, 4096]
colors = ['black', 'white']

for data in ['amazon-computers', 'flickr', 'yelp']:
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    box_data = []
    for bs in batch_sizes:
        # read file
        for alg in ['gat', 'gcn']:
            # data
            file_path = os.path.join('batch_more_inference_memory', '_'.join([alg, data, str(bs)]) + '.csv')
            res = pd.read_csv(file_path, index_col=0)['memory'].values.tolist()
            box_data.append(list(map(lambda x: x/(1024*1024), res)))
    
    if box_data == []:
        continue
    print(box_data)
    bp = ax.boxplot(box_data, patch_artist=True)
    numBoxes = len(batch_sizes) * 2
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        if i % 2 == 1:
            plt.setp(bp['medians'][i], color='red')
            plt.setp(bp['boxes'][i], color='red')
            plt.setp(bp['boxes'][i], facecolor=colors[1])
            plt.setp(bp['fliers'][i], markeredgecolor='red')
            # https://matplotlib.org/stable/gallery/statistics/boxplot.html#sphx-glr-gallery-statistics-boxplot-py
        else:
            plt.setp(bp['boxes'][i], facecolor=colors[0])
    
    ax.set_title(data, fontsize=16)
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Inference Memory Usage (MB)', fontsize=13)
    ax.set_xticks([1.5, 3.5, 5.5])
    ax.set_xticklabels(batch_sizes, fontsize=13)

    legend_colors = [Patch(facecolor='black', edgecolor='black'), Patch(facecolor='white', edgecolor='red')]
    ax.legend(legend_colors, ['GCN', 'GAT'], fontsize=14)
    fig.savefig(f'exp3_thesis_figs/exp_inference_sampling_memory_fluctuation_{data}.png')
    plt.close()