import os, re, sys
import numpy as np
import pandas as pd

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr", "com-amazon"]
models = ["gcn", "ggnn", "gat", "gaan"]
gcn_ggnn_hds = ["16", "32", "64", "128", "256", "512", "1024", "2048"]
gat_hds = ["8", "16", "32", "64", "128", "256"]
gat_heads = ["1", "2", "4", "8", "16"]
gaan_hds = ["16", "32", "64", "128", "256", "512", "1024", "2048"]
gaan_ds = ["8", "16", "32", "64", "128", "256"]
gaan_heads = ["1", "2", "4", "8", "16"]


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
                             