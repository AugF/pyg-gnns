from collections import defaultdict as dd
import numpy as np
import pickle as pkl

np.random.seed(1)

# 1. setting parameter
dir = "com-amazon"
file_name = "com-amazon.ungraph.txt"

# get id-map: id real_node
id_map = {}
with open(dir + "/id-map.txt") as f:
    for line in f.readlines():
        t = line.strip().split('\t')
        id_map[int(t[1])] = int(t[0])

# get graph
graph = dd(list)
with open(dir + "/" + file_name) as f:
    for line in f.readlines():
        if not line.startswith("#"):
            t = line.strip().split('\t')
            a, b = id_map[int(t[0])], id_map[int(t[1])]
            graph[a].append(b)

with open(dir + "/ind.{}.graph".format(dir), "wb") as f:
    pkl.dump(graph, f)
