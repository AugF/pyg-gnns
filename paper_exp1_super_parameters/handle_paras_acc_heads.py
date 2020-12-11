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
    if not os.path.exists("acc_res"):
        os.makedirs("acc_res")
    model = "gat"
    dir_config = "dir_gat_acc"
    
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
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line:
                            acc = int(float(match_line.group(1)) * 100) / 10000
                            break
                df_heads[data].append(acc)
        df = pd.DataFrame(df_heads, index=gat_heads)
        df.columns = datasets_map
        df.to_csv(f"acc_res/{model}_heads_{hd}.csv")


def pics_gat(dir_in="acc_res", dir_out="acc_res"):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = ['1', '2', '4', '8', '16']
    xlabels =  [r"#Head ($d_{head}$=1)", r"#Head ($d_{head}$=2)", r"#Head ($d_{head}$=4)"]
    
    for i, hd in enumerate(["1", "2", "4"]):
        df = pd.read_csv(dir_in + f"/gat_heads_{hd}.csv", index_col=0)
        df.index = xticklabels
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabels[i])
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            df[c].plot(ax=ax, marker=markers[j], label=c, rot=0)
        ax.legend()
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_prefix + f"gat_heads_{hd}.png")
        plt.close()

save_acc_to_csv()
pics_gat()