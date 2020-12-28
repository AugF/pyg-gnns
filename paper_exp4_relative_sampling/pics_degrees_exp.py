import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use("ggplot")
plt.rcParams['font.size'] = 12

graph_path = "amp_graph.npy"
cluster_path = "amp_cluster_90.npy"
graphsage_path = "amp_graphsage_459.npy"

paths = [graph_path, cluster_path, graphsage_path]
names = ['Original Graph', 'Cluster Sampling', 'Neighbor Sampling']

dir_in = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp4_relative_sampling/batch_degrees_distribution"
dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/exp_supplement"

def get_degrees_counts(path):
    # 1. 统计出每个节点的度数
    in_degrees = {}
    out_degrees = {}

    graph = np.load(path)
    for i in graph[0, :]:
        if i not in in_degrees.keys():
            in_degrees[i] = 1
        else:
            in_degrees[i] += 1

    for i in graph[1, :]:
        if i not in out_degrees.keys():
            out_degrees[i] = 1
        else:
            out_degrees[i] += 1

    degrees_counts = {}
    for i in in_degrees.values():
        if i not in degrees_counts.keys():
            degrees_counts[i] = 1
        else:
            degrees_counts[i] += 1

    for i in out_degrees.values():
        if i not in degrees_counts.keys():
            degrees_counts[i] = 1
        else:
            degrees_counts[i] += 1
    return degrees_counts


fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

#ax.set_yscale("symlog", basey=2)
#ax.set_xscale("symlog", basey=2)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(ymin=0.5, ymax=1e4)
ax.set_xlim(xmin=0.5, xmax=1e4)
ax.set_xlabel("Degrees", fontsize=16)
ax.set_ylabel("Numbers", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

markers = 'oD^'
colors = 'rgb'
for i, path in enumerate(paths):
    degrees_counts = get_degrees_counts(dir_in + "/" + path)
    xs = list(degrees_counts.keys())
    ys = [degrees_counts[d] for d in xs]
    ax.scatter(xs, ys, color=colors[i], label=names[i], marker=markers[i])

ax.legend(fontsize=14)

fig.savefig(dir_out + "/exp_sampling_minibatch_degrees_distribution_amazon-photo.png")
fig.savefig(dir_out + "/exp_sampling_minibatch_degrees_distribution_amazon-photo.pdf")