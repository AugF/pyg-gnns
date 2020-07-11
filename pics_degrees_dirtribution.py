import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use("ggplot")
plt.rcParams['font.size'] = 12

graph_path = "amp_graph.npy"
cluster_path = "amp_cluster.npy"
graphsage_path = "amp_graphsage.npy"

paths = [graph_path, cluster_path, graphsage_path]
names = ['Original Graph', 'Cluster-GCN Sampling', 'GraphSAGE Sampling']

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


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Degrees")
ax.set_ylabel("Numbers")

markers='oD^'
colors = 'rgb'
for i, path in enumerate(paths):
    degrees_counts = get_degrees_counts(path)
    xs = list(degrees_counts.keys())
    ys = [degrees_counts[d] for d in xs]
    ax.scatter(xs, ys, color=colors[i], label=names[i], marker=markers[i])

ax.legend()

fig.savefig("exp_sampling_minibatch_degrees_distribution_amazon-photo.png")

