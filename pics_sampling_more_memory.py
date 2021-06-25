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
re_percents = [0.01, 0.03, 0.06, 0.10, 0.25, 0.50]
re_labels = ['1%', '3%', '6%', '10%', '25%', '50%']
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "batch_more_memory_figs")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'reddit', 'yelp']

batch_sizes = [1024, 2048, 4096]
colors = ['black', 'white']

for mode in ['cluster', 'graphsage']:
    # for data in datasets:
    for data in ['amazon-computers', 'flickr', 'yelp']:
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        box_data = []
        if mode == 'cluster':
            batch_sizes = [40, 60, 80]
        else:
            batch_sizes = [1024, 2048, 4096]
        for i, bs in enumerate(batch_sizes):
            # read file
            for alg in ['gat', 'gcn']:
                # data
                file_path = os.path.join('batch_more_memory', '_'.join([mode, alg, data, str(bs)]) + '.csv')
                res = pd.read_csv(file_path, index_col=0)['memory'].values.tolist()
                box_data.append(list(map(lambda x: x/(1024*1024), res)))

        if box_data == []:
            continue
        print(mode, data)
        print(batch_sizes)
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
        ax.set_ylabel('Training Memory Usage (MB)', fontsize=13)
        ax.set_xticks([1.5, 3.5, 5.5])
        ax.set_xticklabels(batch_sizes, fontsize=13)

        legend_colors = [Patch(facecolor='black', edgecolor='black'), Patch(facecolor='white', edgecolor='red')]
        if data in ['amazon-computers', 'yelp'] and mode == 'graphsage':
            ax.legend(legend_colors, ['GCN', 'GAT'], fontsize=12, loc='lower left')
        else:
            ax.legend(legend_colors, ['GCN', 'GAT'], fontsize=14)
        fig.savefig(f'exp3_thesis_figs/exp_sampling_memory_fluctuation_{data}_{mode}.png')
        plt.close()