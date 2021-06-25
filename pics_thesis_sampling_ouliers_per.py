import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.font_manager import _rebuild
_rebuild() 

# plt.style.use("grayscale")
base_size = 12
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = base_size

def box_plot_outliers(data_ser, box_scale=3):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，取3; 默认whis=1.5
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    :return:
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    # 下阈值
    val_low = data_ser.quantile(0.25) - iqr*0.5
    # 上阈值
    val_up = data_ser.quantile(0.75) + iqr*0.5
    # 异常值
    # outlier = data_ser[(data_ser < val_low) | (data_ser > val_up)]
    up_outlier = data_ser[(data_ser > val_up)]
    # 正常值
    # normal_value = data_ser[(data_ser > val_low) & (data_ser < val_up)]
    return up_outlier

dir_in = '/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/batch_more_memory'
dir_infer_in = '/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/batch_more_inference_memory'
dir_out = '/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/sampling_outliers_per'

markers = 'oD^sdp'
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 1)), (0, (5, 5))]
labels = ['GCN', 'GGNN', 'GAT', 'GaAN']
re_bs = [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]
x_labels = [f'{int(100*i)}%' for i in re_bs]
infer_bs = [1024, 2048, 4096, 8192, 16384]
locations = [-1.5, -0.5, 0.5, 1.5]
width = 0.2

# training
for mode in ['cluster', 'graphsage']:
    for data in ['amazon-computers', 'amazon-photo',  'flickr', 'reddit']:
        dd_data = defaultdict(list)
        for model in ['gcn', 'ggnn', 'gat', 'gaan']:
            for rs in re_bs:
                file_name = mode + '_' + model + '_' + data + '_' + str(rs) + '.csv'
                real_path = dir_in + '/' +file_name
                if not os.path.exists(real_path):
                    dd_data[model].append(None)
                    continue
                memory = pd.read_csv(real_path)['memory']
                up_outlier = box_plot_outliers(memory, box_scale=3)
                print(up_outlier, len(memory), len(up_outlier) / len(memory))
                dd_data[model].append(100 * len(up_outlier) / len(memory))
        
        dd_data = pd.DataFrame(dd_data)
        dd_data.to_csv(dir_out + f'/exp_sampling_outliers_per_{mode}_{data}.csv')
        print(dd_data)
        x = np.arange(len(re_bs))
        
        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        for i, c in enumerate(dd_data.columns):
            ax.plot(x, dd_data[c], linestyle=linestyles[i], marker=markers[i], markersize=7, label=labels[i])

        ax.set_title(data, fontsize=base_size + 2)
        ax.set_ylabel('异常点百分比 (%)', fontsize=base_size + 1)
        ax.set_xlabel('相对批规模', fontsize=base_size + 2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        fig.savefig(dir_out + f'/exp_sampling_outliers_per_{mode}_{data}.png')

# inference
for data in ['amazon-computers', 'amazon-photo', 'flickr', 'yelp']:
    dd_data = defaultdict(list)
    for model in ['gcn', 'ggnn', 'gat', 'gaan']:
        for rs in infer_bs:
            file_name = model + '_' + data + '_' + str(rs) + '.csv'
            real_path = dir_infer_in + '/' +file_name
            if not os.path.exists(real_path):
                dd_data[model].append(None)
                continue
            memory = pd.read_csv(real_path)['memory']
            up_outlier = box_plot_outliers(memory, box_scale=3)
            print(up_outlier, len(memory), len(up_outlier) / len(memory))
            dd_data[model].append(100 * len(up_outlier) / len(memory))
    
    dd_data = pd.DataFrame(dd_data)
    dd_data.to_csv(dir_out + f'/exp_sampling_outliers_per_infer_{data}.csv')
    print(dd_data)
    x = np.arange(len(infer_bs))
    
    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    for i, c in enumerate(dd_data.columns):
        ax.plot(x, dd_data[c], linestyle=linestyles[i], marker=markers[i], markersize=7, label=labels[i])

    ax.set_title(data, fontsize=base_size + 2)
    ax.set_ylabel('异常点百分比 (%)', fontsize=base_size+2)
    ax.set_xlabel('批规模', fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(infer_bs)
    ax.legend()
    fig.savefig(dir_out + f'/exp_sampling_outliers_per_infer_{data}.png')