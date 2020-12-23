import re
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

modes = ["cluster", "graphsage"]
algs = ["gcn", "ggnn", "gat", "gaan"]
datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr"]
cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625]
}

datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

xticklabels = [1, 3, 6, 10, 25, 50]

dir_in = "log_fix_time_new"
dir_out = "res_fix_time_new"

if not os.path.exists(dir_out + "/res_csv"):
    os.makedirs(dir_out + "/res_csv")

if not os.path.exists(dir_out + "/res_fig"):
    os.makedirs(dir_out + "/res_fig")
        
def get_full_acc():
    df = {}
    for alg in algs:
        df[alg] = {}
        for data in datasets:
            final_acc = 0.0
            print("\n")
            for mode in modes:
                file_name = dir_in + "/" + mode + "_" + alg + "_" + data + "_full.log"
                print(file_name)
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Batch: .*,.*best_test_acc: (.*), cur_use_time:.*", line)
                        if match_line:
                            acc = float(match_line.group(1))
                final_acc += acc
            final_acc /= 2
            df[alg][data] = final_acc
    pd.DataFrame(df).to_csv(dir_out + "/res_csv/full_acc.csv")


for mode in modes:
    for alg in algs:
        df_accs= {}
        for data in datasets:
            df_accs[data] = []
            # 将数据存储到csv文件
            if mode == "cluster":
                for i, cs in enumerate(cluster_batchs):
                    file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(cs)]) + ".log")
                    if not os.path.exists(file_path):
                        flag = True
                        break
                    print(file_path)
                    acc = None
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time:.*s", line)
                            if match_line:
                                acc = float(match_line.group(1))
                    df_accs[data].append(acc)
            elif mode == "graphsage":
                for i, gs in enumerate(graphsage_batchs[data]):
                    file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(gs)]) + ".log")
                    if not os.path.exists(file_path):
                        flag = True
                        break
                    acc = None
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time:.*", line)
                            if match_line:
                                acc = float(match_line.group(1))
                    df_accs[data].append(acc)
        df = pd.DataFrame(df_accs)
        df.index = xticklabels
        df.to_csv(dir_out + "/res_csv/" + alg + "_" + mode + ".csv")
        
        fig, ax = plt.subplots()
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Relative Batch Size (%)")
        
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            ax.plot(xticklabels, df[c], marker=markers[i], label=datasets_maps[c])
        ax.legend()
        fig.savefig(dir_out + "/res_fig/" + alg + "_" + mode + ".png")        
            
