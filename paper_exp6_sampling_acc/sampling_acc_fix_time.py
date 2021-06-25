"""
sampling_batch_acc_cum_fix_time:
采样的对比实验代码
"""
import os
import time
import numpy as np
import pandas as pd

config_paras = pd.read_csv("/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp5_paras_acc/acc_res/max_acc.csv", index_col=0)

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
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

fix_times = {
    'pubmed': 1000,
    'coauthor-physics': 2000,
    'flickr': 2000,
    'amazon-photo': 4000,
    'amazon-computers': 4000
}

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log_fix_time_12_22")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for i, mode in enumerate(['graphsage']):
    sh_commands = []
    for lr in [0.1]:
        for alg in ["gcn"]:
            for data in ["amazon-photo", "amazon-computers"]:
                paras = str(config_paras.loc[datasets_maps[data], alg]).split('_')
                if alg in ["gcn", "ggnn"]:
                    config_str = f"--hidden_dims {paras[0]}"
                elif alg == "gat":
                    config_str = f"--heads {paras[0]} --head_dims {paras[1]}"
                elif alg == "gaan":
                    config_str = f"--heads {paras[0]} --d_v {paras[1]} --d_a {paras[1]} --d_m {paras[1]} --hidden_dims {paras[2]}"
                if mode == 'cluster':
                    cmd = "python -u /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_batch_acc_cum_fix_time.py --epochs 10000 --lr {} --mode {} --model {} --data {} --device cuda:{} --batch_partitions {} {} --fix_time {} >>{} 2>&1"
                    for cs in cluster_batchs:
                        file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(cs)]) + '.log')
                        if os.path.exists(file_path):
                            continue
                        print(file_path)
                        sh_commands.append(cmd.format(str(lr), mode, alg, data, i, str(cs), config_str, fix_times[data], file_path))
                else:
                    cmd = "python -u /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_batch_acc_cum_fix_time.py --epochs 10000 --lr {} --mode {} --model {} --data {} --device cuda:{} --batch_size {} {} --fix_time {} >>{} 2>&1"
                    for gs in graphsage_batchs[data]:
                        file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(gs)]) + '.log')
                        if os.path.exists(file_path):
                            continue
                        print(file_path)
                        sh_commands.append(cmd.format(str(lr), mode, alg, data, i, str(gs), config_str, fix_times[data], file_path))
    with open("sh_" + mode + "_fix_time_12_22.sh", "w") as f:
        for sh in sh_commands:
            f.write(sh + '\n')