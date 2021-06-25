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

# os.chdir("early_stopping1")
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
                val_loss, train_acc, val_acc, test_acc = [], [], [], []
                acc = None
                cnt = 0
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r".*val_loss: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
                        if match_line and cnt < 1000:
                            val_loss.append(float(match_line.group(1)))
                            train_acc.append(float(match_line.group(2)))
                            val_acc.append(float(match_line.group(3)))
                            test_acc.append(float(match_line.group(4)))
                            cnt += 1
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 4))
                labels = ["val_loss", "train_acc", "val_acc", "test_acc"]
                accs = [val_loss, train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_{hd}.png")
                plt.close()
        elif model == "gat":
            dir_config = "dir_gat_acc"
            for hd in gat_hds:
                file_name = f"{dir_config}/config0_{model}_{data}_4_{hd}.log"
                if not os.path.exists(file_name):
                    continue   
                val_loss, train_acc, val_acc, test_acc = [], [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r".*val_loss: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
                        if match_line and cnt < 1000:
                            val_loss.append(float(match_line.group(1)))
                            train_acc.append(float(match_line.group(2)))
                            val_acc.append(float(match_line.group(3)))
                            test_acc.append(float(match_line.group(4)))
                            cnt += 1
                assert len(val_loss) == len(train_acc) and len(val_acc) == len(test_acc) and len(train_acc) == len(val_acc)
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 4))
                labels = ["val_loss", "train_acc", "val_acc", "test_acc"]
                accs = [val_loss, train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_4_{hd}.png")
                plt.close()
            # heads
            for h in gat_heads:
                file_name = f"{dir_config}/config0_{model}_{data}_{h}_32.log"
                if not os.path.exists(file_name):
                    continue   
                val_loss, train_acc, val_acc, test_acc = [], [], [], []
                cnt = 0
                acc = None               
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r".*val_loss: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
                        if match_line and cnt < 1000:
                            val_loss.append(float(match_line.group(1)))
                            train_acc.append(float(match_line.group(2)))
                            val_acc.append(float(match_line.group(3)))
                            test_acc.append(float(match_line.group(4)))
                            cnt += 1
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 4))
                labels = ["val_loss", "train_acc", "val_acc", "test_acc"]
                accs = [val_loss, train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_{h}_32.png")
                plt.close()
        elif model == "gaan":
            dir_config = "dir_gaan_acc"
            # hds
            for hd in gaan_hds:
                file_name = f"{dir_config}/config0_{model}_{data}_4_32_{hd}.log"
                if not os.path.exists(file_name):
                    continue   
                val_loss, train_acc, val_acc, test_acc = [], [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r".*val_loss: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
                        if match_line and cnt < 1000:
                            val_loss.append(float(match_line.group(1)))
                            train_acc.append(float(match_line.group(2)))
                            val_acc.append(float(match_line.group(3)))
                            test_acc.append(float(match_line.group(4)))
                            cnt += 1
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 4))
                labels = ["val_loss", "train_acc", "val_acc", "test_acc"]
                accs = [val_loss, train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_4_32_{hd}.png")
                plt.close()
            # ds
            for d in gaan_ds:
                file_name = f"{dir_config}/config0_{model}_{data}_4_{d}_64.log"
                if not os.path.exists(file_name):
                    continue
                val_loss, train_acc, val_acc, test_acc = [], [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r".*val_loss: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
                        if match_line and cnt < 1000:
                            val_loss.append(float(match_line.group(1)))
                            train_acc.append(float(match_line.group(2)))
                            val_acc.append(float(match_line.group(3)))
                            test_acc.append(float(match_line.group(4)))
                            cnt += 1
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 4))
                labels = ["val_loss", "train_acc", "val_acc", "test_acc"]
                accs = [val_loss, train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_4_{d}_64.png")
                plt.close()
            # heads
            for h in gaan_heads:
                file_name = f"{dir_config}/config0_{model}_{data}_{h}_32_64.log"
                if not os.path.exists(file_name):
                    continue
                val_loss, train_acc, val_acc, test_acc = [], [], [], []
                cnt = 0
                acc = None
                with open(file_name) as f:
                    for line in f:
                        match_line = re.match(r".*val_loss: (.*), Train: (.*), Val: (.*), Test: (.*)", line)
                        if match_line and cnt < 1000:
                            val_loss.append(float(match_line.group(1)))
                            train_acc.append(float(match_line.group(2)))
                            val_acc.append(float(match_line.group(3)))
                            test_acc.append(float(match_line.group(4)))
                            cnt += 1
                fig, ax = plt.subplots()
                x = range(len(train_acc))
                category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, 4))
                labels = ["val_loss", "train_acc", "val_acc", "test_acc"]
                accs = [val_loss, train_acc, val_acc, test_acc]
                for i, c in enumerate(category_colors):
                    ax.plot(x, accs[i], color=c, label=labels[i])
                ax.legend()
                fig.tight_layout() 
                fig.savefig(f"acc_fig/{model}/{data}/config0_{model}_{data}_{h}_32_64.png")
                plt.close()