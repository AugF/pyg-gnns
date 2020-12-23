import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr"]
# datasets = ["amazon-computers"]
modes = ["cluster", "graphsage"]
algs = ["gcn", "ggnn", "gat", "gaan"]
# algs = ["gat"]

cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625]
}

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

dir_in = "log_fix_epoch"
dir_out = "res_fix_epoch"

for data in datasets:
    if not os.path.exists(dir_out + "/" + data):
        os.makedirs(dir_out + "/" + data)
        
"""
目标：绘制不同BatchSize下，精度随Batch Nums的变化结果

数据收集：
算法, 采样方法，数据集，time/acc

绘制图像：
算法+采样方法+time/acc
df.index: 图像的横坐标
df.columns: 图像中的不同的线(不同的BatchSize)
"""
for data in datasets:
    for alg in algs:
        for mode in modes:
            print(data, alg, mode)
            flag = False
            # 将数据存储到csv文件
            df_times, df_accs = {}, {}
            if mode == "cluster":
                for i, cs in enumerate(cluster_batchs):
                    file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(cs)]) + ".log")
                    if not os.path.exists(file_path):
                        flag = True
                        break
                    df_times[xticklabels[i]], df_accs[xticklabels[i]] = [], []
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                            if match_line:
                                df_accs[xticklabels[i]].append(float(match_line.group(1)))
                                df_times[xticklabels[i]].append(float(match_line.group(2)))
                    if len(df_times[xticklabels[i]]) != 100:
                        flag = True
                        break           
            elif mode == "graphsage":
                for i, gs in enumerate(graphsage_batchs[data]):
                    file_path = os.path.join(dir_in, '_'.join([mode, alg, data, str(gs)]) + ".log")
                    df_times[xticklabels[i]], df_accs[xticklabels[i]] = [], []
                    # 读取日志文件，获取bs_times, bs_accs: dims=100
                    if not os.path.exists(file_path):
                        flag = True
                        break
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                            if match_line:
                                df_accs[xticklabels[i]].append(float(match_line.group(1)))
                                df_times[xticklabels[i]].append(float(match_line.group(2)))
                    if len(df_times[xticklabels[i]]) != 100:
                        flag = True
                        break

            if flag: continue

            # 获取full
            file_path = os.path.join("batch_acc_cum_fix_batch", '_'.join([mode, alg, data]) + "_full.log")
            # 读取日志文件，获取bs_times, bs_accs: dims=100
            with open(file_path) as f:
                df_times['FULL'], df_accs['FULL'] = [], []
                for line in f:
                    match_line = re.match(r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                    if match_line:
                        df_accs['FULL'].append(float(match_line.group(1)))
                        df_times['FULL'].append(float(match_line.group(2)))

            # 画精度关于时间的图像
            fig, ax = plt.subplots()
            ax.set_ylabel('Accucary')
            ax.set_xlabel("use time(s)")
            
            category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, len(xticklabels)))   
            
            for key, c in zip(xticklabels, category_colors):
                ax.plot(df_times[key], df_accs[key], color=c, label=key)
                
            ax.legend()
            fig.tight_layout() # 防止重叠
            fig.savefig(dir_out + "/"  + data + "/" + alg + "_" + mode + "_accs_times.png")
            plt.close()    
            

    