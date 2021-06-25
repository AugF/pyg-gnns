"""
preprocess.py
将com-amazon和com-lj的数据处理为以下格式:
（参考：https://github.com/GraphSAINT/GraphSAINT）
- adj_full.npz
- id_map.json
- class_map.json
- feats.npy
- role.json
"""
import numpy as np
import scipy.sparse as sp
import json
import os.path as osp


def get_adj_and_map(raw_dir, name, classes):
    # 1. get id_map.json and adj_full.npz
    ungraph_file = osp.join(raw_dir, name + ".ungraph.txt")
    cmty_file = osp.join(raw_dir, name + ".all.cmty.txt")
    id_map = {}
    nodes = 0
    edges = 0
    row = []
    col = []
    with open(ungraph_file) as f:
        for line in f.readlines():
            if not line.startswith("#"):
                t = line.strip().split('\t')
                row.append(t[0]);
                col.append(t[1])
                if t[0] not in id_map.keys():
                    id_map[t[0]] = nodes;
                    nodes += 1
                if t[1] not in id_map.keys():
                    id_map[t[1]] = nodes;
                    nodes += 1
                edges += 1

    row = [id_map[i] for i in row]
    col = [id_map[i] for i in col]

    f = sp.csr_matrix(([1] * edges, (row, col)), shape=(nodes, nodes))
    np.savez(raw_dir + "/adj_full", data=f.data, indptr=f.indptr, indices=f.indices, shape=f.shape)

    with open(raw_dir + "/id_map.json", "w") as f:
        json.dump(id_map, f)

    # 2. get class_map.json
    class_map = {}
    cnt = 0
    with open(cmty_file) as f:
        for line in f.readlines():
            t = line.strip().split('\t')
            for i in t:
                class_map[id_map[i]] = cnt % classes
            cnt += 1

    with open(raw_dir + "/class_map.json", "w") as f:
        json.dump(class_map, f)

    return nodes, edges


# 2. get feats.npy
def get_feats(raw_dir, nodes, features, seed=1):
    np.random.seed(seed)
    feats = np.random.randn(nodes, features)
    np.save(raw_dir + "/feats", feats)


# 3. get role.json: random split
def get_role(raw_dir, nodes, tr, va, seed=1):
    np.random.seed(seed)
    idx = np.arange(nodes)
    np.random.shuffle(idx)
    tr, va = int(nodes * tr), int(nodes * (tr + va))
    role = {'tr': idx[: tr].tolist(),
            'va': idx[tr: va].tolist(),
            'te': idx[va:].tolist()
            }

    with open(raw_dir + "/role.json", "w") as f:
        json.dump(role, f)


def generate_com_lj():
    name = "com-lj"
    features, classes, tr, va = 32, 10, 0.50, 0.25
    raw_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'epochs', name, "raw")

    print("get_adj_and_map...")
    nodes, edges = get_adj_and_map(raw_dir=raw_dir,
                                   name=name,
                                   classes=classes)

    print("nodes: ", nodes, "edges: ", edges)
    print("get_feats...")
    get_feats(raw_dir=raw_dir, nodes=nodes, features=features)
    print("get role...")
    get_role(raw_dir=raw_dir, nodes=nodes, tr=tr, va=va)


def generate_com_amazon():
    name = "com-amazon"
    features, classes, tr, va = 32, 10, 0.50, 0.25
    raw_dir = osp.join("/home/wangzhaokang/wangyunpan/gnns-project/datasets", "com-amazon", "raw")

    print("get_adj_and_map...")
    nodes, edges = get_adj_and_map(raw_dir=raw_dir,
                                   name=name,
                                   classes=classes)

    print("nodes: ", nodes, "edges: ", edges)
    print("get_feats...")
    get_feats(raw_dir=raw_dir, nodes=nodes, features=features)
    print("get role...")
    get_role(raw_dir=raw_dir, nodes=nodes, tr=tr, va=va)


def gen_role():
    datasets = ['amazon-computers', 'amazon-photo', 'coauthor-physics']
    nodes = [13752, 7650, 34493]
    base_path = "/home/wangzhaokang/wangyunpan/gnns-project/datasets"
    
    for data, ns in zip(datasets, nodes):
        raw_dir = base_path + "/" + data + "/raw"
        get_role(raw_dir, ns, tr=0.50, va=0.25, seed=1)
    
if __name__ == '__main__':
    gen_role()
