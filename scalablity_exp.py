import numpy as np
#import snap # in linux, need py35, py36 or py37
import json
import os
import scipy.sparse as sp
# from utils import small_datasets, small_nodes
# https://snap.stanford.edu/snappy/doc/reference/GenRMat.html
# directed graph

def input_dense_feature_exp(datasets, nodes, dir_in=".", seed=1):
    dims = [16, 32, 64, 128, 256, 512]
    np.random.seed(seed)
    for i, name in enumerate(datasets):
        for d in dims:
          feats = np.random.randn(nodes[i], d)
          np.save(dir_in + "/data/feats_x/" + name + "_" + str(d) + "_feats", feats)


def input_sparse_feature_exp(datasets, nodes, seed=1):
    # 1. input feature dimension
    np.random.seed(seed)
    print("begin features ratio exp..")
    # 1.1 dims=500, ratio=0.05, 0.1, 0.2, 0.5
    ratios = [0.05, 0.1, 0.2, 0.5]
    for i, name in enumerate(datasets):
        for r in ratios:  #
            feats = np.random.randn(nodes[i], 500)
            feats = np.where(feats <= r, 0, 1)
            np.save("data/feats_x/" + name + "_500_" + str(int(r * 100)) + "_feats", feats)

    print("begin features dims exp..")
    # 1.2 ratio=0.2, dims=250, 500, 750, 1000, 1250
    dims = [250, 750, 1000, 1250]
    for i, name in enumerate(datasets):
        for d in dims:  #
            feats = np.random.randn(nodes[i], d)
            feats = np.where(feats <= 0.2, 0, 1)
            np.save("data/feats_x/" + name + "_" + str(d) + "_20_feats", feats)


# 2. graph scalablity
def gen_graph(raw_dir, nodes, edges, features=32, classes=10, tr=0.70, va=0.15, seed=1):
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
    # 2.1 degree=25, n=1k, 25k, 50k, 75k, 100k
    ns = [1000, 25000, 50000, 75000, 100000]
    names = ['1k', '25k', '50k', '75k', '100k']
    degree_fix = 25
    for i, nodes in enumerate(ns):
        edges = nodes * degree_fix
        print("nodes={}, edges={}".format(nodes, edges))
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "epochs/graph_" + names[i] + "_25/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        print("get adj_full.npz...")
        edges_list = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            edges_list.append((r, c))
            edges_list.append((c, r))
        
        edges_list = set(edges_list)
        row = [i[0] for i in edges_list]
        col = [i[1] for i in edges_list]
        print(edges, len(row))
        f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected, edges*2
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges)

    # 2.2 n=500k, degree=10, 25, 50, 75, 100
    degrees = [10, 50, 75, 100]  # 25 omit
    nodes = 50000
    for d in degrees:
        edges = nodes * d
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "data/graph_50k_" + str(d) + "/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        print("get adj_full.npz...")
        # gen adj_full
        edges_list = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            edges_list.append((r, c))
            edges_list.append((c, r))
        
        edges_list = set(edges_list)
        row = [i[0] for i in edges_list]
        col = [i[1] for i in edges_list]
        print(edges, len(row))
        f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected, edges*2
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges)

def gen_real_degrees_exp(seed=1):
    Rnd = snap.TRnd()
    degrees=[3, 6, 10, 15, 20, 25, 30, 50]
    nodes = 50000
    for ds in degrees:
        edges = nodes * ds
        print("nodes={}, edges={}".format(nodes, edges))
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "/home/wangzhaokang/wangyunpan/gnns-project/datasets/graph_50k_" + str(ds) + "/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        else:
            continue
        print("get adj_full.npz...")
        # 1. gen adj_full
        edges_list = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            edges_list.append((r, c))
            edges_list.append((c, r))
        
        edges_list = set(edges_list) # 保证真实的数据集，每条边应该只出现一次
        row = [i[0] for i in edges_list]
        col = [i[1] for i in edges_list]
        print(edges, len(row))
        f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges)


def gen_real_edges_memory(seed=1):
    Rnd = snap.TRnd()
    # 2.1 degree=25, n=1k, 25k, 50k, 75k, 100k
    ns = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
    names = ['1k', '5k', '10k', '20k', '30k', '40k', '50k']
    edges = 500000
    for i, nodes in enumerate(ns):
        print("nodes={}, edges={}".format(nodes, edges))
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "/home/wangzhaokang/wangyunpan/gnns-project/datasets/graph_" + names[i] + "_500k/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        else:
            continue
        print("get adj_full.npz...")
        # 1. gen adj_full
        edges_list = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            edges_list.append((r, c))
            edges_list.append((c, r))
        
        edges_list = set(edges_list)
        row = [i[0] for i in edges_list]
        col = [i[1] for i in edges_list]
        print(edges, len(row))
        f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges)

 
def gen_min_edges_graph(seed=1):
    Rnd = snap.TRnd()
    # 2.1 degree=25, n=1k, 25k, 50k, 75k, 100k
    ns = [500, 1000, 2000, 4000, 6000, 8000, 10000, 15000]
    names = ['05k', '1k', '2k', '4k', '6k', '8k', '10k', '15k']
    degrees = 4
    for j in range(50):
        for i, nodes in enumerate(ns):
            edges = nodes * degrees
            print("nodes={}, edges={}".format(nodes, edges))
            graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
            raw_dir = "/home/wangzhaokang/wangyunpan/gnns-project/datasets/graph_" + names[i] + "_4_" + str(j) + "/raw"
            print(raw_dir)
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
            else:
                continue
            print("get adj_full.npz...")
            # 1. gen adj_full
            edges_list = []
            for e in graph.Edges():
                r, c = e.GetSrcNId(), e.GetDstNId()
                edges_list.append((r, c))
                edges_list.append((c, r))
            
            edges_list = set(edges_list)
            row = [i[0] for i in edges_list]
            col = [i[1] for i in edges_list]
            print(edges, len(row))
            f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected
            np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
            gen_graph(raw_dir, nodes, edges)
 
 
def gen_real_degrees_memory(seed=1):
    Rnd = snap.TRnd()
    degrees = [2, 5, 10, 15, 25, 20, 30, 40, 50, 70]  # 25 omit
    nodes = 10000
    for d in degrees:
        edges = nodes * d
        graph = snap.GenRMat(nodes, edges, .6, .1, .15, Rnd)
        raw_dir = "/home/wangzhaokang/wangyunpan/gnns-project/datasets/graph_10k_" + str(d) + "/raw"
        print(raw_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        else:
            continue
        print("get adj_full.npz...")
        # gen adj_full
        edges_list = []
        for e in graph.Edges():
            r, c = e.GetSrcNId(), e.GetDstNId()
            edges_list.append((r, c))
            edges_list.append((c, r))
        
        edges_list = set(edges_list)
        row = [i[0] for i in edges_list]
        col = [i[1] for i in edges_list]
        print(edges, len(row))
        f = sp.csr_matrix(([1] * len(row), (row, col)), shape=(nodes, nodes)) # directed -> undirected
        np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)
        gen_graph(raw_dir, nodes, edges)

# gen_real_degrees_exp()
# gen_real_edges_memory()
# gen_real_degrees_memory()
# gen_min_edges_graph(seed=1)
input_dense_feature_exp(["com-amazon"], [334863], dir_in="/home/wangzhaokang/wangyunpan/gnns-project/datasets", seed=1)
