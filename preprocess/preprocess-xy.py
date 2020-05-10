from collections import defaultdict as dd
import numpy as np
import pickle as pkl

np.random.seed(1)

# 1. setting parameter
features = 32
nodes = 3997962
dir = "com-lj"
file_id = "com-lj.all.cmty.txt"

# get id-map: id real_node
id_map = {}
with open(dir + "/id-map.txt") as f:
    for line in f.readlines():
        t = line.strip().split('\t')
        id_map[int(t[1])] = int(t[0])

print("get features")
#get features
allx = np.zeros((nodes, features))
for i in range(nodes):
    for j in range(features):
        allx[i][j] = np.random.randint(0, 2)

with open(dir + "/ind.{}.allx".format(dir), "wb") as f:
    pkl.dump(allx, f)

#get lables
print("get lables")
cnt = 0
ally = np.zeros((nodes, 10))
with open(dir + "/" + file_id) as f:
    for line in f.readlines():
        t = line.strip().split('\t')
        for i in t:
            ally[id_map[int(i)]][cnt % 10] = 1
        cnt = cnt + 1

with open(dir + "/ind.{}.ally".format(dir), "wb") as f:
    pkl.dump(ally, f)

