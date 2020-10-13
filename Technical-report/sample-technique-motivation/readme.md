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
| --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-products | 2,449,029 | 61,859,140 | 50.5 | 有 | 无 | Multi-class type | full, NS(NeigborSampling), Cluster, GraphSAINT |SAGE aggr, GAT(缺full, GraphSAINT, 暂不考虑) |
| ogbn-mag | 1,939,743 | 21,111,007 | 21.7 | 有 | 有 | Multi-class type | full, NS(NeigborSampling), Cluster, GraphSAINT | RGCN aggr |
| ogbl-citation | 2,927,963 | 30,561,187 | 20.7| 有 | 无 | Link Prediction |  full, NS(NeigborSampling), Cluster, GraphSAINT | GCN aggr(缺NS) |
