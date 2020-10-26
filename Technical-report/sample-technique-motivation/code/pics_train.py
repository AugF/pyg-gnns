'''
该文件是为了画train, val, test随epoch变化的图像
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

batch_size = {
    'ogbn-mag': {
        'cluster_gcn': [50, 150, 300, 500, 1250, 2500],
        'neighbor_sampling': [19397, 58192, 116384, 193974, 484935, 969871, 20000] # ['1%', '3%', '6%', '10%', '25%', '50%'] + 默认batch_size
    }, 
    'ogbn-products': {
        'cluster_gcn': [32, 50, 150, 300, 500, 1250, 2500],
        'neighbor_sampling': [24490, 73470, 146941, 244902, 612257, 1224514, 20000]
    }
}

base_path = "/home/wangzhaokang/wangyunpan/pyg-gnns/Technical-report/sample-technique-motivation/code"

def handle(data, alg, para):
    para = str(para)
    os.chdir(base_path + "/" + data)
    file = alg + "_" + para + ".npy"
    if not os.path.exists(file):
        return
    x = np.load(file)
    df = pd.DataFrame(x[0], columns=['train', 'val', 'test'])
    fig, ax = plt.subplots()
    plt.ylim(0, 1)
    df.plot(ax=ax, kind='line')
    fig.savefig(alg + "_" + para + ".png")
    plt.close()


for data in ['ogbn-mag', 'ogbn-products']:
    for alg in ['cluster_gcn', 'neighbor_sampling', 'graph_saint']:
        if alg == 'cluster_gcn':
            for para in batch_size[data][alg]:
                handle(data, alg, para)
        else:
            for para in batch_size[data]['neighbor_sampling']:
                handle(data, alg, para)
