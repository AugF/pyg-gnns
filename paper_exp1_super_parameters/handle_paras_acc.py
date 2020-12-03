import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr", "com-amazon"]
models = ["gcn", "ggnn", "gat", "gaan"]
gcn_ggnn_hds = ["16", "32", "64", "128", "256", "512", "1024", "2048"]
gat_hds = ["8", "16", "32", "64", "128", "256"]
gat_heads = ["1", "2", "4", "8", "16"]
gaan_hds = ["16", "32", "64", "128", "256", "512", "1024", "2048"]
gaan_ds = ["8", "16", "32", "64", "128", "256"]
gaan_heads = ["1", "2", "4", "8", "16"]

datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

def save_acc_to_csv(): # 将统计得到的acc结果保存在csv文件中 
    if not os.path.exists("acc_res"):
        os.makedirs("acc_res")
    for model in models:
        if model == "gcn" or model == "ggnn":
            dir_config = "dir_gcn_ggnn_acc"
            # hds
            df_hds = {}
            for data in datasets:
                df_hds[data] = []
                for hd in gcn_ggnn_hds:
                    file_name = f"{dir_config}/config0_{model}_{data}_{hd}.log"
                    if not os.path.exists(file_name):
                        df_hds[data].append(None)
                        continue                    
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test: (.*)", line)
                            if match_line:
                                acc = match_line.group(1)
                                break
                    df_hds[data].append(acc)
            pd.DataFrame(df_hds, index=gcn_ggnn_hds).to_csv(f"acc_res/{model}_hds.csv")
        elif model == "gat":
            dir_config = "dir_gat_acc"
            # hds
            df_hds = {}
            for data in datasets:
                df_hds[data] = []
                for hd in gat_hds:
                    file_name = f"{dir_config}/config0_{model}_{data}_4_{hd}.log"
                    if not os.path.exists(file_name):
                        df_hds[data].append(None)
                        continue   
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test: (.*)", line)
                            if match_line:
                                acc = match_line.group(1)
                                break
                    df_hds[data].append(acc)
            pd.DataFrame(df_hds, index=gat_hds).to_csv(f"acc_res/{model}_hds.csv")
            
            # heads
            df_heads = {}
            for data in datasets:
                df_heads[data] = []
                for h in gat_heads:
                    file_name = f"{dir_config}/config0_{model}_{data}_{h}_32.log"
                    if not os.path.exists(file_name):
                        df_heads[data].append(None)
                        continue   
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test: (.*)", line)
                            if match_line:
                                acc = match_line.group(1)
                                break
                    df_heads[data].append(acc)
            pd.DataFrame(df_heads, index=gat_heads).to_csv(f"acc_res/{model}_heads.csv")
        elif model == "gaan":
            dir_config = "dir_gaan_acc"
            # hds
            df_hds = {}
            for data in datasets:
                df_hds[data] = []
                for hd in gaan_hds:
                    file_name = f"{dir_config}/config0_{model}_{data}_4_32_{hd}.log"
                    if not os.path.exists(file_name):
                        df_hds[data].append(None)
                        continue   
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test: (.*)", line)
                            if match_line:
                                acc = match_line.group(1)
                                break
                    df_hds[data].append(acc)
            pd.DataFrame(df_hds, index=gaan_hds).to_csv(f"acc_res/{model}_hds.csv")
            
            # ds
            df_ds = {}
            for data in datasets:
                df_ds[data] = []
                for d in gaan_ds:
                    file_name = f"{dir_config}/config0_{model}_{data}_4_{d}_64.log"
                    if not os.path.exists(file_name):
                        df_ds[data].append(None)
                        break
                    acc = None
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test: (.*)", line)
                            if match_line:
                                acc = float(match_line.group(1))
                                break
                    df_ds[data].append(acc)
                
            pd.DataFrame(df_ds, index=gaan_ds).to_csv(f"acc_res/{model}_ds.csv")
                
            # heads
            df_heads = {}
            for data in datasets:
                df_heads[data] = []
                for h in gaan_heads:
                    file_name = f"{dir_config}/config0_{model}_{data}_{h}_32_64.log"
                    if not os.path.exists(file_name):
                        df_heads[data].append(None)
                        break
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test: (.*)", line)
                            if match_line:
                                acc = match_line.group(1)
                                break
                    df_heads[data].append(acc)
            pd.DataFrame(df_heads, index=gaan_heads).to_csv(f"acc_res/{model}_heads.csv")


def pics_gcn_ggnn(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = ['16', '32', '64', '128', '256', '512', '1024', '2048']
    xlabel = "Hidden Dimension"
    algs = ['gcn', 'ggnn']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

    for alg in algs:
        df = pd.read_csv(dir_in + "/" + alg + "_hds.csv", index_col=0)
        df.index = xticklabels
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabel)
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            df[c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
        ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + ".png")
        plt.close()


def pics_gat(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabels =  [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"]
    
    for i, mode in enumerate(['hds', 'heads']):
        df = pd.read_csv(dir_in + "/gat_" + mode + ".csv", index_col=0)
        df.index = xticklabels[i]
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabels[i])
        ax.set_xticks(list(range(len(xticklabels[i]))))
        ax.set_xticklabels(xticklabels[i])
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
        ax.legend()
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_prefix + "gat_" + mode + ".png")
        plt.close()


def pics_gaan(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
    
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], ['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabels = [r"Hidden Vector Dimension $dim(\mathbf{h}^1)$ (#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$(#Head=4, $dim(\mathbf{h}^1)$=64)",  r"#Head ($dim(\mathbf{h}^1)$=64, $d_a=d_v=d_m$=32)"]
   
    for i, mode in enumerate(['hds', 'ds', 'heads']):
        df = pd.read_csv(dir_in + "/gaan_" + mode + ".csv", index_col=0)
        df.index = xticklabels[i]
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabels[i])
        ax.set_xticks(list(range(len(xticklabels[i]))))
        ax.set_xticklabels(xticklabels[i])
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
        ax.legend()
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_prefix + "gaan_" + mode + ".png")
        plt.close()
        

pics_gat()
pics_gaan()