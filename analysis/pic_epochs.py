import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
plt.style.use('ggplot')


# for epochs pic
def pic_lines(file_name):
    df = pd.read_csv("epochs/" + file_name + ".csv", index_col=0)

    ax = plt.gca()
    # 设置固定刻度值
    # x_major_locator = MultipleLocator(1)
    # y_major_locator = MultipleLocator(100)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    ax.set_ylabel("ms")
    ax.set_xlabel("epoch")

    ylim = (0, 800)
    if 'reddit-2' in file_name:
        ylim = (800, 1600)

    df.plot(kind='line', title=file_name, rot=0, ax=ax, ylim=ylim, marker='*')
    plt.show()
    fig = ax.get_figure()
    fig.savefig("epochs/lines/" + file_name + '.png')
    del ax


def pic_violin(file_name):
    df = pd.read_csv("epochs/" + file_name + ".csv", index_col=0)

    mean_x = df.values.mean(axis=0).reshape(1, -1)

    # get percent
    all_data = 100 * df.values / mean_x - 100
    labels = df.columns.to_list()

    ax = plt.gca()
    ax.set_title(file_name)
    ax.set_ylabel("percent")
    ax.set_xlabel("algorithm")
    ax.violinplot(all_data, showmeans=False, showmedians=True)
    set_axis_style(ax, labels)
    plt.show()
    fig = ax.get_figure()
    fig.savefig('epochs/violin/' + file_name + '.png')
    del ax


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


if __name__ == '__main__':
    for i in [0, 2]:
        for name in ['flickr', 'com-amazon', 'reddit', 'reddit-2', 'com-lj']:
            file_name = 'config{}_{}'.format(i, name)
            if os.path.isfile('epochs/' + file_name + '.csv'):
                pic_lines(file_name)
                pic_violin(file_name)











