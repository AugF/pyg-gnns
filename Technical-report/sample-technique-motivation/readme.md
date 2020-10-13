参考benchmark: [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://ogb.stanford.edu/docs/home/), 已经被NIPS2020接收

## Benchmark

### 数据集
![](figs/table2.png)

![](figs/table3.png)

### 学习任务

根据Task分类，可以分为：
- node property prediction
    - [ogbn-products](https://ogb.stanford.edu/docs/leader_nodeprop/)
    ![](figs/node-ogbn-products.png)
        - GraphSAINT(SAGE aggr)
        - ClusterGCN(SAGE aggr)
        - NeighborSampling(SAGE aggr)
        - Full-batch GraphSAGE
        - GAT+NeighborSampling
        - Cluster-GAT
    - [ogbn-mag](https://ogb.stanford.edu/docs/leader_nodeprop/)
    ![](figs/node-ogbn-mag.png)
        - HGT(LADIES Sample)
        - GraphSAINT(R-GCN aggr)
        - NeighborSampling(R-GCN aggr)
        - Full-batch(R-GCN aggr)
        - ClusterGCN(R-GCN aggr)
- link prboperty prediction
    - [ogbl-citation](https://ogb.stanford.edu/docs/leader_linkprop/)
    ![](figs/link-ogbl-citation.png)
        - Full-batch GCN
        - Full-batch GraphSAGE
        - NeighborSampling(SAGE aggr)
        - ClusterGCN(GCN aggr)
        - GraphSAINT(GCN aggr)
- graph property prediction(无合适)


## Motivation验证实验

选取ogbn-products, ogbn-mag, ogbl-citaton三个数据集，数据集基本信息为

| 数据集 | 点数 | 边数 | 平均度数 | 点特征数 | 边特征数 | 学习任务 | 采样算法 | 基本模型 |
| --- | --- | --- | --- | --- | --- | --- | --- | -- |
| ogbn-products | 2,449,029 | 61,859,140 | 50.5 | 有 | 无 | Multi-class type | full, NS(NeigborSampling), Cluster, GraphSAINT |SAGE aggr |
| ogbn-mag | 1,939,743 | 21,111,007 | 21.7 | 有 | 有 | Multi-class type | full, NS(NeigborSampling), Cluster, GraphSAINT | RGCN aggr |
| ogbl-citation | 2,927,963 | 30,561,187 | 20.7| 有 | 无 | Link Prediction |  full, NS(NeigborSampling), Cluster, GraphSAINT | GCN aggr(缺NS) |

### 实验环境

Tesla T4 16GB * 2
[参数Tesla T4 vs V100](https://blog.csdn.net/tony_vip/article/details/105658715)

- [x] 代码审查
- [x] 可执行测试
    > ogdb-products和ogbl-citations下Full-batch内存不够
- [ ] 正确性测试, 并记录训练参数
- [ ] 参数收集, 划分比例为1%, 3%, 6%, 10%, 25%, 50%
    - [ ] 收集Cluster-GCN的num_partitions
    - [ ] 收集NS, GraphSAINT的Batch Size