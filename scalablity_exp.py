import numpy as np
import snap # in linux, need py35, py36 or py37
import json
import os
import scipy.sparse as sp
# https://snap.stanford.edu/snappy/doc/reference/GenRMat.html
# directed graph

def input_dense_feature_exp(seed=1):
    datasets = ['flickr', 'com-amazon']
    nodes = [89250, 334863]
    dims = [16, 32, 64, 128, 256, 512]
    np.random.seed(seed)
    for i, name in enumerate(datasets):
        for d in dims:
          feats = np.random.randn(nodes[i], d)
          np.save("data/" + name + "_" + str(d) + "/raw/feats", feats) 
    

def input_sparse_feature_exp(seed=1):
    # 1. input feature dimension
    datasets = ['flickr', 'com-amazon']
    nodes = [89250, 334863]

    np.random.seed(seed)
    print("begin features ratio exp..")
    # 1.1 dims=500, ratio=0.05, 0.1, 0.2, 0.5
    ratios = [0.05, 0.1, 0.2, 0.5]
    for i, name in enumerate(datasets):
        for r in ratios:  #
            feats = np.random.randn(nodes[i], 500)
            feats = np.where(feats <= r, 0, 1)
            np.save("data/" + name + "_500_" + str(int(r * 100)) + "/raw/feats", feats)

    print("begin features dims exp..")
    # 1.2 ratio=0.2, dims=250, 500, 750, 1000, 1250
    dims = [250, 750, 1000, 1250]
    for i, name in enumerate(datasets):
        for d in dims:  #
            feats = np.random.randn(nodes[i], d)
            feats = np.where(feats <= 0.2, 0, 1)
            np.save("data/" + name + "_" + str(d) + "_20/raw/feats", feats)


# 2. graph scalablity
def gen_graph(raw_dir, nodes, edges, features=32, classes=10, tr=0.50, va=0.25, seed=1):
    np.random.seed(seed)
    print("get class_map.json...")
    # 1. class_map.json
    class_map = {i: np.random.randint(0, classes) for i in range(nodes)}
    with open(raw_dir + "/class_map.json", "w") as f:
        json.dump(class_map, f)
    
    print("get role.json...")
    # 2. role.json: tr, va, te
    idx = np.arange(nodes)
    np.random.shuffle(idx)
    tr, va = int(nodes * tr), int(nodes * (tr + va))
    role = {'tr': idx[: tr].tolist(),
            'va': idx[tr: va].tolist(),
            'te': idx[va:].tolist()
            }

    with open(raw_dir + "/role.json", "w") as f:
        json.dump(role, f)
    
    print("get feats.npy...")
    # 3. feats.npy
    feats = np.random.randn(nodes, features)
    np.save(raw_dir + "/feats", feats)


def graph_scale_exp(seed=1):
    Rnd = snap.TRnd()
    # 2.1 degree=25, n=10k, 50k, 100k, 500k
    ns = [1000000, 5000000]
    names = ['1m', '5m'] # [100k, 500k, 1m, 5m]
    degree_fix = 25
    for i, nodes in enumerate(ns):
        edges = nodes * degree_fix
        print("nodes={}, edges={}".format(nodes, edges))
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "data/graph_" + names[i] + "_25/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        print("get adj_full.npz...")
        # 1. gen adj_full
        row = []
        col = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            row.append(r)
            col.append(c)
            row.append(c)
            col.append(r)
        f = sp.csr_matrix(([1] * 2 * edges, (row, col)), shape=(nodes, nodes)) # directed -> undirected, edges*2
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges, seed)

    # 2.2 n=500k, degree=10, 25, 50, 75, 100
    degrees = [10, 50, 75, 100]  # 25 omit
    nodes = 500000
    for d in degrees:
        edges = nodes * d
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "data/graph_500k_" + str(d) + "/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        print("get adj_full.npz...")
        # gen adj_full
        row = []
        col = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            row.append(r)
            col.append(c)
            row.append(c)
            col.append(r)
        f = sp.csr_matrix(([1] * 2 * edges, (row, col)), shape=(nodes, nodes)) # directed -> undirected, edges*2
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges, seed)


if __name__ == '__main__':
    print("begin input feature experiment...")
    input_dense_feature_exp()






