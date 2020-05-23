"""
预处理文件：
dict: c
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('ggplot')


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


def pic_stages(name, columns):
    """
    :param name: eg: config0_flickr
    :param columns: eg: ['gcn', 'ggnn', 'gat', 'gaan']
    :return:
    """
    arr = np.recfromtxt("stages/" + name + ".txt")
    x = np.array(arr).reshape(-1, 3)
    sum_x = np.sum(x, axis=1).reshape(-1, 1)
    out = 100 * x / sum_x
    res = {}
    for i, x in enumerate(columns):
        res[x] = list(out[i])
    fig, ax = survey(res, ['forward', 'backward', 'eval'])
    ax.set_title(name + " dataset")
    ax.set_xlabel("%")
    ax.set_ylabel("algorithm")
    fig.savefig('stages/' + name + '.png')
    plt.show()


def pic_stages_by_algorithm(file_name):
    df = pd.read_csv("stages/" + file_name + ".csv", index_col=0)
    sum_x = df.values.sum(axis=1).reshape(-1, 1)
    out = 100 * df.values / sum_x
    res = {}
    for i, x in enumerate(df.index):
        res[x] = list(out[i])
    fig, ax = survey(res, df.keys().tolist())
    ax.set_title("algorithm " + file_name)
    ax.set_xlabel("%")
    ax.set_ylabel("algorithm")
    fig.savefig('stages/' + file_name + '.png')
    plt.show()
    del ax


if __name__ == '__main__':
    for file_name in ['gcn', 'ggnn', 'gat', 'gaan']:
        pic_stages_by_algorithm(file_name)
