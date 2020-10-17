## pyg-gnns

基于pytorch-geometric的图神经网络评测

### 1. description
#### 1.1 dataset
dataset
- cora
- flickr
- com-amazon
- reddit
- com-lj
    
dataset格式(N为边数, F为特征数， C为类别数）
- adj_full.npz: a sparse matrix in csr format, ['indptr', 'indices', 'data', 'shape'], N * N
- feats.npy: a numpy array, N * F
- role.json: a dictionary of three keys. Key 'tr' corresponds to the list of all training node indices. Key va corresponds to the list of all validation node indices. Key te corresponds to the list of all test node indices. Note that in the raw data, nodes may have string-type ID. You would need to re-assign numerical ID (0 to N-1) to the nodes, so that you can index into the matrices of adj, features and class labels.
- class_map.json:a dictionary of length N. Each key is a node index, and each value is either a length C binary list (for multi-class classification) or an integer scalar (0 to C-1, for single-class classification)

#### 1.2 model
1. gcn
2. ggnn
3. gat
4. gaan

### 2. Usage
`
python main.py --help   
`

一些修改：

对于多标签分类问题

1. yelp引入的关于多标签分类的问题

- 单标签分类：F.log_softmax(x, 1)
- 多标签分类：


### 3. 一些问题的更新

关于/data/wangzhaokang/wangyunpan目录被删除了
关于gcn, ggnn, gat的最初实验结果被删除了


    
