import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr"]
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
    'com-amazon': 'cam'
}

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']

dir_in = "log_fix_time_new"
dir_out = "sampling_acc_time_figs"

if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    
max_accs = pd.read_csv("/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp5_paras_acc/acc_res/alg_acc.csv", index_col=0)

for mode in ["cluster"]:
    for alg in ["gcn"]:
        for data in ["amazon-computers"]:
            # 将数据存储到csv文件
            df_times, df_accs = {}, {}
            if mode == "cluster":
                for i, cs in enumerate(cluster_batchs):
                    file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(cs)]) + ".log")
                    df_times[xticklabels[i]], df_accs[xticklabels[i]] = [], []
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                            if match_line:
                                df_accs[xticklabels[i]].append(float(match_line.group(1)))
                                df_times[xticklabels[i]].append(float(match_line.group(2)))
            elif mode == "graphsage":
                for i, gs in enumerate(graphsage_batchs[data]):
                    file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(gs)]) + ".log")
                    df_times[xticklabels[i]], df_accs[xticklabels[i]] = [], []
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                            if match_line:
                                df_accs[xticklabels[i]].append(float(match_line.group(1)))
                                df_times[xticklabels[i]].append(float(match_line.group(2)))
            
            fig, ax = plt.subplots()
            category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, len(xticklabels)))   
            
            min_len = np.inf
            for key, c in zip(xticklabels, category_colors):
                ax.plot(df_times[key], df_accs[key], color=c, label=key)
                min_len = min(min_len, len(df_times[key]))
                
            ax.set_ylabel('Accucary')
            ax.set_xlabel("Time (s)")
            ax.plot(range(min_len), [max_accs.loc[datasets_maps[data], alg]] * min_len, linestyle='--')
            ax.legend()
            fig.tight_layout() # 防止重叠
            fig.savefig(dir_out + "/"  + mode + "_" + alg + "_" + data + "_accs_times.png")
            plt.close()    
            

