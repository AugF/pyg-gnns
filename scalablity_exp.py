import numpy as np
import snap # in linux, need py35, py36 or py37
import json
import os
import scipy.sparse as sp

# https://snap.stanford.edu/snappy/doc/reference/GenRMat.html
# directed graph


def input_feature_exp(seed=1):
    # 1. input feature dimension
    datasets = ['flickr', 'com-amazon']
    nodes = [89250, 334863]

    np.random.seed(seed)
    # 1.1 dims=500, ratio=0.05, 0.1, 0.2, 0.5
    ratios = [0.05, 0.1, 0.2, 0.5]
    for i, name in enumerate(datasets):
        for r in ratios:  #
            feats = np.random.randn(nodes[i], 500)
            feats = np.where(feats <= r, 0, 1)
            np.save("data/" + name + "_500_" + str(int(r * 100)) + "/raw/feat", feats)

    # 1.2 ratio=0.2, dims=250, 500, 750, 1000, 1250
    dims = [250, 750, 1000, 1250]
    for i, name in enumerate(datasets):
        for d in dims:  #
            feats = np.random.randn(nodes[i], d)
            feats = np.where(feats <= 0.2, 0, 1)
            np.save("data/" + name + "_" + str(d) + "_20/raw/feat", feats)


# 2. graph scalablity
def gen_graph(graph, raw_dir, nodes, edges, features=32, classes=10, tr=0.50, va=0.25, seed=1):
    np.random.seed(seed)
    # 1. adj_full.npz
    row = col = []
    for e in graph.Edges():
        r, c = e.GetSrcNId(), e.GetDstNId()
        row.append(r)
        col.append(c)
        row.append(c)
        col.append(r)
    f = sp.csr_matrix(([1] * edges, (row, col)), shape=(nodes, nodes))
    np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
    # 2. id_map.json
    id_map = {i: i for i in range(nodes)}
    with open(raw_dir + "/id_map.json", "w") as f:
        json.dump(id_map, f)

    # 3. class_map.json
    class_map = {i: np.random.randint(0, classes) for i in range(nodes)}
    with open(raw_dir + "/id_map.json", "w") as f:
        json.dump(class_map, f)

    # 4. role.json: tr, va, te
    idx = np.arange(nodes)
    np.random.shuffle(idx)
    tr, va = int(nodes * tr), int(nodes * (tr + va))
    role = {'tr': idx[: tr].tolist(),
            'va': idx[tr: va].tolist(),
            'te': idx[va:].tolist()
            }

    with open(raw_dir + "/role.json", "w") as f:
        json.dump(role, f)

    # feats.npy
    feats = np.random.randn(nodes, features)
    np.save(raw_dir + "/feats", feats)


def graph_scale_exp(seed=1):
    Rnd = snap.TRnd()
    # 2.1 degree=25, n=100k, 500k, 1m, 5m
    ns = [100000, 500000, 1000000, 5000000]
    names = ['100k', '500k', '1m', '5m']
    degree_fix = 25
    for i, n in enumerate(ns):
        graph = snap.GenRMat(n, n * degree_fix, .6, .1, .15, Rnd)
        raw_dir = "data/graph_" + names[i] + "_25/raw"
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        gen_graph(graph, raw_dir, nodes=n, edges=n * degree_fix)

    # 2.2 n=500k, degree=10, 25, 50, 75, 100
    degrees = [10, 50, 75, 100]  # 25 omit
    nodes = 500000
    for d in degrees:
        graph = snap.GenRMat(nodes, nodes * d, .6, .1, .15, Rnd)
        raw_dir = "data/graph_500k_" + str(d) + "/raw"
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        gen_graph(graph, raw_dir, nodes=500000, edges=nodes * d, seed=seed)


if __name__ == '__main__':
    input_feature_exp()
    graph_scale_exp()




