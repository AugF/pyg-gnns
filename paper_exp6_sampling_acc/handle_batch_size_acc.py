import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr"]

algs = ["gcn", "ggnn", "gat", "gaan"]
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

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

acc_full = pd.read_csv("../paper_exp1_super_parameters/early_stopping2/acc_res/alg_acc.csv", index_col=0)

dir_in = "res1/batch_acc"
dir_out = "res1/acc_res"
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

for mode in modes:
    for alg in algs:
        # 目标：看算法本身的精度随batch_size的变化的结果
        # df_data = {} 
        # for data in datasets:
        #     df_data[data] = []
        #     if mode == "cluster":
        #         for cs in cluster_batchs:
        #             file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(cs)]) + '.log')
        #             acc = None
        #             if not os.path.exists(file_path):
        #                 df_data[data].append(acc)
        #                 continue
        #             with open(file_path) as f:
        #                 for line in f:
        #                     match_line = re.match("   Final Test: (.*) ± .*", line)
        #                     if match_line:
        #                         acc = int(float(match_line.group(1)) * 100) / 10000 # 精度汇报，报告截断后的数
        #                         break
        #             df_data[data].append(acc)
        #     else:
        #         for gs in graphsage_batchs[data]:
        #             file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(gs)]) + '.log')
        #             acc = None
        #             if not os.path.exists(file_path):
        #                 df_data[data].append(acc)
        #                 continue
        #             with open(file_path) as f:
        #                 for line in f:
        #                     match_line = re.match("   Final Test: (.*) ± .*", line)
        #                     if match_line:
        #                         acc = int(float(match_line.group(1)) * 100) / 10000 # 精度汇报，报告截断后的数
        #                         break
        #             df_data[data].append(acc)
        #     df_data[data].append(acc_full.loc[datasets_maps[data], alg])
        # df = pd.DataFrame(df_data, index=xticklabels)
        df = pd.read_csv(dir_out + "/" + alg + "_" + mode + ".csv", index_col=0)

        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy')
        ax.set_xlabel("Relative Batch Size(%)")
        ax.set_ylim(0, 1)
        
        ax.set_xticks(list(range(len(xticklabels)))) # todo: 汇报精度随BatchSize的变化时，将横坐标整理为标准坐标`ax.plot(numbers, y)`, 画折线图
        ax.set_xticklabels(xticklabels)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[j], label=c, rot=0)
        ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + alg + "_" + mode + ".png")
        plt.close()