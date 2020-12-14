import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

models = ["gcn", "ggnn", "gat", "gaan"]
datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr"]
gcn_ggnn_hds = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"]
gat_hds = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
gat_heads = ["1", "2", "4", "8", "16"]
gaan_hds = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"]
gaan_ds = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
gaan_heads = ["1", "2", "4", "8", "16"]

os.chdir("early_stopping1")
for model in models:
    for data in datasets:
        if not os.path.exists(f"acc_fig/{model}/{data}"):
            os.makedirs(f"acc_fig/{model}/{data}")
        print(model, data)
        if model in ["gcn", "ggnn"]:
            dir_config = "dir_gcn_ggnn_acc"
            for hd in gcn_ggnn_hds:
                file_name = f"{dir_config}/config0_{model}_{data}_{hd}.log"
                if not os.path.exists(file_name):
                    continue   
                print(file_name)
                train_acc, val_acc, test_acc = [], [], []
                acc = None
                cnt = 0
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Epoch: .*, Train: (.*), Val: (.*), Test: (.*)", line)
                        match_final_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line and cnt < 1000:
                            train_acc.append(float(match_line.group(1)))
                            val_acc.append(float(match_line.group(2)))
                            test_acc.append(float(match_line.group(3)))
                            cnt += 1
                        if match_final_line:
                            acc = int(float(match_final_line.group(1)) * 100) / 10000
                print("Final Test", acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 3))
                labels = ["train_acc", "val_acc", "test_acc"]
                accs = [train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_{hd}.png")
        elif model == "gat":
            dir_config = "dir_gat_acc"
            for hd in gat_hds:
                file_name = f"{dir_config}/config0_{model}_{data}_4_{hd}.log"
                if not os.path.exists(file_name):
                    continue   
                train_acc, val_acc, test_acc = [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Epoch: .*, Train: (.*), Val: (.*), Test: (.*)", line)
                        match_final_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line and cnt < 1000:
                            train_acc.append(float(match_line.group(1)))
                            val_acc.append(float(match_line.group(2)))
                            test_acc.append(float(match_line.group(3)))
                            cnt += 1
                        if match_final_line:
                            acc = int(float(match_final_line.group(1)) * 100) / 10000
                print("Final Test", acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 3))
                labels = ["train_acc", "val_acc", "test_acc"]
                accs = [train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_4_{hd}.png")
            
            # heads
            for h in gat_heads:
                file_name = f"{dir_config}/config0_{model}_{data}_{h}_32.log"
                if not os.path.exists(file_name):
                    continue   
                train_acc, val_acc, test_acc = [], [], []
                cnt = 0
                acc = None               
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Epoch: .*, Train: (.*), Val: (.*), Test: (.*)", line)
                        match_final_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line and cnt < 1000:
                            train_acc.append(float(match_line.group(1)))
                            val_acc.append(float(match_line.group(2)))
                            test_acc.append(float(match_line.group(3)))
                            cnt += 1
                        if match_final_line:
                            acc = int(float(match_final_line.group(1)) * 100) / 10000
                print("Final Test", acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 3))
                labels = ["train_acc", "val_acc", "test_acc"]
                accs = [train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_{h}_32.png")
            
        elif model == "gaan":
            dir_config = "dir_gaan_acc"
            # hds
            for hd in gaan_hds:
                file_name = f"{dir_config}/config0_{model}_{data}_4_32_{hd}.log"
                if not os.path.exists(file_name):
                    continue   
                train_acc, val_acc, test_acc = [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Epoch: .*, Train: (.*), Val: (.*), Test: (.*)", line)
                        match_final_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line and cnt < 1000:
                            train_acc.append(float(match_line.group(1)))
                            val_acc.append(float(match_line.group(2)))
                            test_acc.append(float(match_line.group(3)))
                            cnt += 1
                        if match_final_line:
                            acc = int(float(match_final_line.group(1)) * 100) / 10000
                print("Final Test", acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 3))
                labels = ["train_acc", "val_acc", "test_acc"]
                accs = [train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_4_32_{hd}.png")
            
            # ds
            for d in gaan_ds:
                file_name = f"{dir_config}/config0_{model}_{data}_4_{d}_64.log"
                if not os.path.exists(file_name):
                    continue
                train_acc, val_acc, test_acc = [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Epoch: .*, Train: (.*), Val: (.*), Test: (.*)", line)
                        match_final_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line and cnt < 1000:
                            train_acc.append(float(match_line.group(1)))
                            val_acc.append(float(match_line.group(2)))
                            test_acc.append(float(match_line.group(3)))
                            cnt += 1
                        if match_final_line:
                            acc = int(float(match_final_line.group(1)) * 100) / 10000
                print("Final Test", acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 3))
                labels = ["train_acc", "val_acc", "test_acc"]
                accs = [train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_4_{d}_64.png")
                
            # heads
            for h in gaan_heads:
                file_name = f"{dir_config}/config0_{model}_{data}_{h}_32_64.log"
                if not os.path.exists(file_name):
                    continue
                train_acc, val_acc, test_acc = [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r"Epoch: .*, Train: (.*), Val: (.*), Test: (.*)", line)
                        match_final_line = re.match("   Final Test: (.*) ± .*", line)
                        if match_line and cnt < 1000:
                            train_acc.append(float(match_line.group(1)))
                            val_acc.append(float(match_line.group(2)))
                            test_acc.append(float(match_line.group(3)))
                            cnt += 1
                        if match_final_line:
                            acc = int(float(match_final_line.group(1)) * 100) / 10000
                print("Final Test", acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 3))
                labels = ["train_acc", "val_acc", "test_acc"]
                accs = [train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_{h}_32_64.png")