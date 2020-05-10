from collections import defaultdict as dd
import numpy as np
import json
import pickle as pkl

from scipy import sparse
from networkx.readwrite import json_graph

prefix = "ppi/ppi"

# 1. get id-map.txt:  id: real_id
id_map = json.load(open(prefix + "-id_map.json"))

ids = {int(v): k for k, v in id_map.items}
with open("reddit/id-map.txt", "w") as f:
    for k in sorted(ids.keys()):
        f.write(str(k) + "\t" + ids[k] + "\n")

# 2. get graph
G_data = json.load(open(prefix + "-G.json"))
G = json_graph.node_link_graph(G_data)

graph = dd(list)
for edge in G.edges():
    if not 'val' in G.node[edge[0]] or not 'test' in G.node[edge[0]]: continue
    if not 'val' in G.node[edge[1]] or not 'test' in G.node[edge[1]]: continue
    a, b = id_map[edge[0]], id_map[edge[1]]
    graph[a].append(b)
    graph[b].append(a)

with open("reddit/ind.reddit.graph", "wb") as f:
    pkl.dump(graph, f)

# 3. get allx
feats = np.load(prefix + "-feats.npy")
allx = sparse.csr_matrix(feats)
with open("reddit/ind.reddit.allx", "wb") as f:
    pkl.dump(allx, f)

# 4. get ally
class_map = json.load(open(prefix + "-class_map.json"))

ally = np.zeros((len(class_map), 41))
for k in class_map.keys():
    ally[id_map[k]][int(class_map[k])] = 1

with open("reddit/ind.reddit.ally", "wb") as f:
    pkl.dump(ally, f)
