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
gat_heads = ["1", "2", "4", "8", "16"]

def save_acc_to_csv(): # 将统计得到的acc结果保存在csv文件中 
    dir_config = "dir_gat_heads_exp"
    if not os.path.exists(dir_config):
        os.makedirs(dir_config)
    model = "gat"
    
    for hd in ["1", "2", "4"]:
        # heads
        df_heads = {}
        for data in datasets:
            df_heads[data] = []
            for h in gat_heads:
                file_name = f"{dir_config}/config0_{model}_{data}_{h}_{hd}.log"
                if not os.path.exists(file_name):
                    df_heads[data].append(None)
                    continue   
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match("Final Test Acc: (.*)", line)
                        if match_line:
                            acc = float(match_line.group(1))
                            print(acc)
                df_heads[data].append(acc)
        df = pd.DataFrame(df_heads, index=gat_heads)
        df.columns = datasets_map
        df.to_csv(f"acc_res/{model}_heads_{hd}.csv")


def pics_gat(dir_in="acc_res", dir_out="acc_res"):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = ['1', '2', '4', '8', '16']
    xlabels =  [r"#Head ($d_{head}$=2)", r"#Head ($d_{head}$=4)"]
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 7/2), sharey=True, tight_layout=True)
    for i, hd in enumerate(["2", "4"]):
        df = pd.read_csv(dir_in + f"/gat_heads_{hd}.csv", index_col=0)
        df.index = xticklabels
        ax = axes[i]
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabels[i], fontsize=12)
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels, fontsize=10)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            ax.plot(df.index, df[c], marker=markers[j], label=c)
        ax.legend(ncol=1, fontsize='xx-small')
    fig.tight_layout() 
    fig.savefig(dir_out + "/" + file_prefix + f"gat_small_info.png")
    fig.savefig(dir_out + "/" + file_prefix + f"gat_small_info.pdf")
    plt.close()

# save_acc_to_csv()
pics_gat(dir_in="acc_res", dir_out="paras_figs")