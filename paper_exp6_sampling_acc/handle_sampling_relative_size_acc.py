import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed",
            "amazon-computers", "coauthor-physics", "flickr"]
modes = ["cluster", "graphsage"]
algs = ["gcn", "ggnn", "gat", "gaan"]

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
}

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']
xticks = [1, 3, 6, 10, 25, 50]

dir_in = "log_fix_time_new"
dir_out = "sampling_acc_figs"

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

for mode in modes:
    fig, axes = plt.subplots(2, 2, figsize=(
        7, 7), tight_layout=True)
    cnt = 0
    for alg in algs:
        df_accs = {}
        for data in datasets:
            df_accs[data] = []
            if mode == "cluster":
                for i, cs in enumerate(cluster_batchs):
                    file_path = os.path.join(dir_in, '_'.join(
                        [mode, alg, data, str(cs)]) + ".log")
                    if not os.path.exists(file_path):
                        df_accs[data].append(np.nan)
                        continue
                    print(file_path)
                    test_acc = 0
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(
                                r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                            if match_line:
                                test_acc = max(test_acc, float(match_line.group(1)))
                    df_accs[data].append(np.nan if test_acc == 0 else test_acc)
            elif mode == "graphsage":
                for i, gs in enumerate(graphsage_batchs[data]):
                    file_path = os.path.join(dir_in, '_'.join(
                        [mode, alg, data, str(gs)]) + ".log")
                    if not os.path.exists(file_path):
                        df_accs[data].append(np.nan)
                        continue
                    print(file_path)
                    test_acc = 0
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(
                                r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                            if match_line:
                                test_acc = max(test_acc, float(match_line.group(1)))
                    df_accs[data].append(np.nan if test_acc == 0 else test_acc)

        print(df_accs, cnt // 2, cnt % 2)
        pd.DataFrame(df_accs, index=xticklabels).to_csv(dir_out + "/" + mode + "_" + alg + ".csv")
        # 画图
        ax = axes[cnt // 2][cnt % 2]
        markers = 'oD^sdp'
        for i in range(5):
            ax.plot(xticks, df_accs[datasets[i]], marker=markers[i], label=datasets_maps[datasets[i]])
        ax.set_title(algorithms[alg], loc="right", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel("Relative Batch Size (%)", fontsize=12)
        ax.legend()
        cnt += 1
    fig.savefig(dir_out + "/exp_sampling_relative_batch_size_accuracy_" + mode + ".png")
    fig.savefig(dir_out + "/exp_sampling_relative_batch_size_accuracy_" + mode + ".pdf")
