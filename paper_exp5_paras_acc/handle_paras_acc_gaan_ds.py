import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr", "com-amazon"]
datasets_map = ['amp', 'pub', 'amc', 'cph', 'fli', 'cam']

models = ["gcn", "ggnn", "gat", "gaan"]
gaan_ds = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]

def save_acc_to_csv(dir_work="./"): # 将统计得到的acc结果保存在csv文件中 
    dir_config = "dir_gaan_ds_acc"
    model = "gaan"
    df_ds = {}
    for data in datasets:
        df_ds[data] = []
        for d in gaan_ds:
            file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_1_{d}_8.log"
            if not os.path.exists(file_name):
                df_ds[data].append(None)
                continue
            print(file_name)
            acc = None
            with open(file_name) as f:
                for line in f:
                    match_line = re.match("Final Test Acc: (.*)", line)
                    if match_line:
                        acc = format(float(match_line.group(1)), ".5f")
                        break
            df_ds[data].append(acc)
    df = pd.DataFrame(df_ds, index=gaan_ds)
    df.columns = datasets_map
    df.to_csv(f"{dir_work}/acc_res/{model}_ds.csv")


def pics_gaan_ds(dir_in="acc_res", dir_out="acc_res"):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

    file_prefix = "exp_hyperparameter_on_accuracy_"
    xlabels =  r"$d_a, d_v, d_m$" + "\n" + r"(#Head=4, $dim(\mathbf{h}^1_x)$=64)"
    xticklabels = ['1', '2', '4', '8', '16', '32', '64', '128', '256']
    fig, ax = plt.subplots(figsize=(7/2, 7/2), sharey=True, tight_layout=True)
    
    df = pd.read_csv(dir_in + f"/gaan_ds.csv", index_col=0)
    df.index = xticklabels
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0.4, 1)
    ax.set_xlabel(xlabels, fontsize=12)
    ax.set_xticks(list(range(len(xticklabels))))
    ax.set_xticklabels(xticklabels, fontsize=10)
    markers = 'oD^sdp'
    for j, c in enumerate(df.columns[:-1]):
        ax.plot(df.index, df[c], marker=markers[j], label=c)
        
    ax.legend()
    fig.tight_layout() 
    fig.savefig(dir_out + "/" + file_prefix + f"gaan_ds_small_info.png")
    fig.savefig(dir_out + "/" + file_prefix + f"gaan_ds_small_info.pdf")
    plt.close()

save_acc_to_csv()
pics_gaan_ds()