import matplotlib.pyplot as plt
import numpy as np
import os
import re

plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

sampling_modes = {
    'graphsage': 'GraphSAGE',
    'cluster': 'Cluster-GCN'
}

datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
algs = ['gcn', 'ggnn', 'gat', 'gaan']

def get_data(file_path):
    losses = []
    train_acces = []
    test_acces = []
    with open(file_path) as f:
        for line in f:
            if "full" in file_path:
                acc_match_line = re.match(r"Accuracy: Train: (.*), Val.*Test: (.*)", line)
                loss_match_line = re.match(r"Epoch:.*train_loss: (.*),.*", line)
                if acc_match_line:
                    train_acces.append(float(acc_match_line.group(1)))
                    test_acces.append(float(acc_match_line.group(2)))
                if loss_match_line:
                    losses.append(float(loss_match_line.group(1)))                  
            else:
                if "batch" in file_path:
                    match_line = re.match(r".*Batch.*Loss: (.*), Train: (.*), Val.*test: (.*)", line)
                elif "epoch" in file_path:
                    match_line = re.match(r".*Epoch.*Loss: (.*), Train: (.*), Val.*test: (.*)", line)
                if match_line:
                    losses.append(float(match_line.group(1)))
                    train_acces.append(float(match_line.group(2)))
                    test_acces.append(float(match_line.group(3)))
    # assert len(losses) == len(train_acces) and len(losses) == len(test_acces) 
    if "full" in file_path:
        del losses[0]
    print(len(losses), len(train_acces), len(test_acces))
    
    return np.array([losses, train_acces, test_acces])

def pics(params):
    labels, title, fig_name = params['labels'], params['title'], params['fig_name']
    xlabel, data = params['xlabel'], params['data']
    len_x = data.shape[1]
    x = list(range(1, len_x + 1))
    if xlabel == 'Batch':
        x = [i * 10 for i in x]
    colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, 3))
    markers = 'oD^sdp'

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.set_ylabel("Loss")
    line1, = ax2.plot(x, data[0, :], label=labels[0], color=colors[0], marker=markers[0])
    line2, = ax.plot(x, data[1, :], label=labels[1], color=colors[1], marker=markers[1])
    line3, = ax.plot(x, data[2, :], label=labels[2], color=colors[2], marker=markers[2])
            
    plt.legend(handles=[line1, line2, line3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accucary")
    ax.set_title(title)
    fig.savefig(fig_name)
    plt.close()


def pics_ax(params, ax):
    labels, title, fig_name = params['labels'], params['title'], params['fig_name']
    xlabel, data = params['xlabel'], params['data']
    len_x = data.shape[1]
    x = list(range(1, len_x + 1))
    if xlabel == 'Batch':
        x = [i * 10 for i in x]
    colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, 3))
    markers = 'oD^sdp'

    ax2 = ax.twinx()
    ax2.set_ylabel("Loss")
    line1, = ax2.plot(x, data[0, :], label=labels[0], color=colors[0], marker=markers[0])
    line2, = ax.plot(x, data[1, :], label=labels[1], color=colors[1], marker=markers[1])
    line3, = ax.plot(x, data[2, :], label=labels[2], color=colors[2], marker=markers[2])
            
    plt.legend(handles=[line1, line2, line3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accucary")
    ax.set_title(title)
    return ax


def run(algs, datasets):
    batch_labels = ["Batch Loss", "Train Acc", "Test Acc"]
    epoch_labels = ["Epoch Loss", "Train Acc", "Test Acc"]
    
    for alg in algs:
        for data in datasets:
            plt.figure(figsize=(6 * 3, 4.8 * 2))
            fig = plt.figure(1)
            cnt = 1
            for file in ['cluster_batch', 'cluster_epoch', 'graphsage_batch', 'graphsage_epoch', 'full']:
                npy_file = "sampling_valids/" + alg + "_" + data + "_" + file + ".npy"
                if not os.path.exists(npy_file):
                    log_file = "sampling_valids_log/" + alg + "_" + data + "_" + file + ".log"
                    if not os.path.exists(log_file):
                        continue
                    data_file = get_data(log_file)
                else:
                    data_file = np.load(npy_file, allow_pickle=True)
                print(alg, data, file)
                params = {
                    'labels': batch_labels if 'batch' in file else epoch_labels,
                    'title': alg + "_" + data + "_" + file,
                    'fig_name': "sampling_valids/" + alg + "_" + data + "_" + file + ".png",
                    'xlabel': 'Batch' if 'batch' in file else 'Epoch',
                    'data': data_file
                }
                ax = plt.subplot(3, 2, cnt)
                cnt += 1
                ax = pics_ax(params, ax)
            fig.tight_layout()
            fig.savefig("sampling_valids/" + alg + "_" + data + ".png")
            plt.close()


def pics_losses(data, xlabel, fig_name, title=""):
    x = np.arange(1, data.shape[1] + 1)
    fig, ax = plt.subplots()
    markers = 'oD'
    for i, ax_label in enumerate(['Train Loss', 'Eval Loss']):
        ax.plot(x, data[i, :], label=ax_label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.savefig(fig_name)
    plt.close()
    

if __name__ == "__main__":
    # run(["gcn"], ["pubmed"])
    run(["gcn"], datasets)
    run(algs, ['pubmed'])
        