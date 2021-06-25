import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr"]
models = ["gcn", "ggnn", "gat", "gaan"]
modes = ["cluster", "graphsage"]

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

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']

dir_in = "log_fix_time_new"
dir_out = "res_convergence_validation"

for data in datasets:
    for model in models:
        if not os.path.exists(f"{dir_out}/{dir_in}"):
            os.makedirs(f"{dir_out}/{dir_in}")
        for mode in modes:
            # 将数据存储到csv文件
            if mode == "cluster":
                for i, cs in enumerate(cluster_batchs):
                    file_path = os.path.join(dir_in, '_'.join([mode, model, data, str(cs)]) + ".log")
                    if not os.path.exists(file_path):
                        continue
                    train_acc, val_acc = [], []
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch: .*, train_acc: (.*), val_acc: (.*), best_val_acc:.*", line)
                            if match_line:
                                train_acc.append(float(match_line.group(1)))
                                val_acc.append(float(match_line.group(2)))
                    
                    fig, ax = plt.subplots()
                    x = range(len(train_acc))
                    category_colors = plt.get_cmap('RdYlGn')(
                    np.linspace(0.15, 0.85, 2))
                    labels = ["train_acc", "val_acc"]
                    accs = [train_acc, val_acc]
                    for i, c in enumerate(category_colors):
                        ax.plot(x, accs[i], color=c, label=labels[i])
                    ax.legend()
                    fig.tight_layout() 
                    fig_path = '_'.join([mode, model, data, str(cs)]) + '.png'
                    print(file_path)
                    fig.savefig(f"{dir_out}/{dir_in}/{fig_path}")
                                         
            elif mode == "graphsage":
                for i, gs in enumerate(graphsage_batchs[data]):
                    file_path = os.path.join(dir_in, '_'.join([mode, model, data, str(gs)]) + ".log")
                    if not os.path.exists(file_path):
                        continue
                    train_acc, val_acc = [], []
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch: .*, train_acc: (.*), val_acc: (.*), best_val_acc:.*", line)
                            if match_line:
                                train_acc.append(float(match_line.group(1)))
                                val_acc.append(float(match_line.group(2)))
                    
                    fig, ax = plt.subplots()
                    x = range(len(train_acc))
                    category_colors = plt.get_cmap('RdYlGn')(
                    np.linspace(0.15, 0.85, 2))
                    labels = ["train_acc", "val_acc"]
                    accs = [train_acc, val_acc]
                    for i, c in enumerate(category_colors):
                        ax.plot(x, accs[i], color=c, label=labels[i])
                    ax.legend()
                    fig.tight_layout() 
                    fig_path = '_'.join([mode, model, data, str(gs)]) + '.png'
                    print(file_path)
                    fig.savefig(f"{dir_out}/{dir_in}/{fig_path}")