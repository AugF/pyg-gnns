# coding=utf-8
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def survey(labels, data, category_names, ax=None, color_dark2=False): # stages, layers, steps，算子可以通用
    # labels: 这里是纵轴的坐标; 
    # data: 数据[[], []] len(data[1])=len(category_names); 
    # category_names: 比例的种类
    for i, c in enumerate(category_names):
        if c[0] == '_':
            category_names[i] = c[1:]

    data_cum = data.cumsum(axis=1)
    if color_dark2:
        category_colors = plt.get_cmap('Dark2')(
            np.linspace(0.15, 0.85, data.shape[1]))
    else:
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))       
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.2, 5))
    else:
        fig = None
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.set_xlabel("Proportion (%)")

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, '%.1f' % c, ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

def pics_minibatch_time(file_type="png"):
    dir_out = "results"
    
    ylabel = "Training Time per Batch (ms)"
    xlabel = "Relative Batch Size (%)"

    cluster_batchs = {
        'ogbn-mag': [50, 150, 300, 500, 1250, 2500],
        'ogbn-products': [50, 150, 300, 500, 1250, 2500]
    }

    graphsage_batchs = {
        'ogbn-mag': [19397, 58192, 116384, 193974, 484935, 969871],
        'ogbn-products': [24490, 73470, 146941, 244902, 612257, 1224514]
    }
    
    xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']

    for data in ['ogbn-mag', 'ogbn-products']:
        for sampling_alg in ['cluster_gcn', 'neighbor_sampling', 'graph_saint']:
            if sampling_alg == 'cluster_gcn':
                paras = cluster_batchs[data]
            else:
                paras = graphsage_batchs[data]
            
            fail = False
            df_data = []
            for k, xlabel in enumerate(xticklabels):
                print("para", paras[k], data, sampling_alg)
                file_path = data + "/" + sampling_alg + "_" + str(paras[k]) + ".log"
                if not os.path.exists(file_path): # 文件不存在
                    fail = True 
                    continue 
                print(file_path)
                train_time, sampling_time, to_time = 0.0, 0.0, 0.0
                success = False
                with open(file_path) as f:
                    for line in f:
                        match_line = re.match(r"Avg_sampling_time: (.*)s, Avg_to_time: (.*)s,  Avg_train_time: (.*)s", line)
                        if match_line:
                            sampling_time = float(match_line.group(1))
                            to_time = float(match_line.group(2))
                            train_time = float(match_line.group(3))
                            success = True
                            break
                # 转化为ms
                if not success: break
                df_data.append([sampling_time, to_time, train_time])
                
            if fail: continue #
            df_data = np.array(df_data)
            np.savetxt(dir_out + "/" + data + "_" + sampling_alg + ".csv", df_data)
            df_data = 100 * df_data / df_data.sum(axis=1).reshape(-1, 1)
            print(df_data)
            
            fig, ax = survey(xticklabels[:df_data.shape[0]], df_data, ['Sampling', 'Data Transferring', 'Training'], color_dark2=True)
            ax.set_title(data + "_" + sampling_alg, loc="right")
            ax.set_xlabel("Proportion (%)")
            ax.set_ylabel("Relative Batch Size (%)")
            plt.tight_layout()
            fig.savefig(dir_out + "/" + data + "_" + sampling_alg + "." + file_type) 
            
            
pics_minibatch_time(file_type="png")
# pics_minibatch_time(file_type="pdf")
