---
title: 系统视角下的图神经网络训练过程性能瓶颈分析
author: 王肇康 王云攀
bibliography: gnn-references.bib
citation-style: chinese-gb7714-2005-numeric.csl
css: academy.css
reference-section-title: 参考文献
figPrefix: 图
figureTitle: 图
tblPrefix: 表
tableTitle: 表
secPrefix: 节
subfigGrid: true
---

# 1 绪论

近年来，图神经网络是人工智能领域内的研究热点，在多种任务多个领域下取得了卓越的成果. 这些成功与graph structure相比于grid data structure有更强大的表现能力，以及深度学习端到端强大的学习能力息息相关。随着图神经网络在多个领域取得好的结果，在系统领域也陆续提出了一系列并行或分布式的图神经网络计算系统。这些系统从大量图神经网络中抽象出图神经网络计算模型，并针对计算模型设计了高效的实现。并在实现中使用了大量的性能优化技巧。
1. message-passing通用模型
PyG[1], DGL[2]基于message-passing机制[3]的计算系统，$\hat{h}_i^{l+1}  = \gamma (\hat{h}_i^{ll}, \Sigma_{j \in \mathcal{N}(i)} \phi(\hat{h}_i^l, \hat{h}_j^l, \hat{e}_{ij}^l))$, 将图卷积操作定义边计算操作message function、聚合操作reduction fucnction和点计算操作update function. 该论文对应的代码实现即pytorch-geometric, 基于PyTorch后端的图神经网络的计算框架。DGL也是基于message-passing的编程模型

2. SAGA-NN通用模型
NeuGraph[4]为图神经网络训练提出了SAGA-NN（Scatter-ApplyEdge-Gather-ApplyVertex with Neural Networks）编程模型。SAGA-NN模型将图神经网络中每一层的前向计算划分为4个阶段：Scatter、ApplyEdge、Gather和ApplyVertex。其中ApplyEdge和ApplyVertex阶段执行用户提供的基于神经网络的边特征向量和点特征向量的计算。Scatter和Gather是由NeuGraph系统隐式触发的阶段，这两个阶段为ApplyEdge和ApplyVertex阶段准备数据。在编程时，用户只需利用给定的算子实现ApplyEdge和ApplyVertex函数，并指定Gather方式，即可利用NeuGraph自动地完成GNN的训练。

3. Sample + Aggregate + Combine通用模型
在AliGraph[5]所支持的通用GNN框架中，每一层的GNN被拆解为三个基本算子: Sample, Aggregate和Combine。其中Sample对应于采样，Aggregate进行边计算，Combine对应于点计算。因为AliGraph面对的是实际大规模图数据，所以AliGraph重点放在了图存储，图采样，图计算三个部分。在图存储上，采用了vertex-cut的方式，即不同的边分配到不同的机器上。在图采样上，支持三种采样方式，Traverse: 从一个图分区中采样一批顶点。Neighborhood, 采样某个顶点的1跳或多跳邻域。Negative,生成负采样样本，加速收敛。特别地，Sampler中的权重也允许根据梯度更新。

在这些图神经网络计算系统实现中，各自用了不同的性能优化技巧，然而这些性能优化技巧是否真正解决了GNN训练过程中的性能瓶颈研究还存有疑问。目前来说，对于图神经网络训练的具体性能瓶颈的分析工作很少，最近的工作[6]Architectural Implications of GNNs, 基于SAGA-NN编程模型和DGL计算系统进行实验，作者认为GNN没有固定的性能瓶颈，性能瓶颈会随着数据集和算法的不同而变化，但是该工作中选择的图的阶数都是很低的情况，而且没有对图的规模和GNN的点边的复杂度进行探讨。工作[7]分析了GCN类的算法在inference阶段的特性，同时与经典的图分析算法(PageRank)和基于MLP的经典神经网络的特性进行了对比分析，发现实际图中的顶点度数分布符合幂等律分布的特性，因此缓存高度数的顶点，有可能可以提升硬件的cache的命中率，因为向量化原子访问可以提升aggregation阶段的效率，但是该工作只选取了某个特定的GNN算法，不能很好地表示大部分的GNN的训练分析。

由于图神经网络每层最本质的操作实际上可以概括为aggregate和update两个操作，aggregate操作即收集邻居顶点的信息, 时间和计算开销与图的边数直接相关；update操作进行顶点信息的变换. 时间和计算开销与图的顶点数直接相关。本文在不考虑复合GNN模型的情况下，将3篇综述[8-10]中的大多数GNN模型进行了点计算和边计算计算量的统计。从点边高低四个象限中选择了四个典型算法，GCN, GGNN, GAT, GaAN进行讨论，作为此次选取的典型GNN算法。此外，本文从Performance执行时间分解，Resource Usage GPU显存使用，Scalability三个方面设计指标，
1. Performance执行时间分解： 对于一个深度学习算法来说，执行所花费的总时间主要是由两部分构成的：time per epoch 和 convergence speed(loss reduction) per epoch. 第二部分是由算法本身决定的，不属于本文的讨论范围。所以这里主要是对time per epoch该部分进行了分析。首先，实验验证了每个epochs训练用时是稳定的；其次，由于目前不支持多GPU背景下训练a big graph, 所以此时我们分析了单GPU下Transductive learning下深度学习训练的分析，将time per epoch划分为了forward, backward, evaluation三个阶段的分析；然后，对GNN模型的每层耗时进行了分解；
2. Resource Usage: 这里考虑使用GPU加速情况下的表现，对应指标为GPU显存占比
3. Scalability: 从三个方面，算法的超参数影响，数据扩展性（特征的维度、稀疏性，图的顶点数和阶数），采样技术三个方面进行了探讨。

本次实验使用message-passing机制

[1] Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. (1), 1–9. 
[2] 
[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (n.d.). Neural Message Passing for Quantum Chemistry.
[4] 
[5]
[6]
[7]
[8]
[9]
[10]

# 2 图神经网络

## 2.1 图神经网络的通用结构

1. graph neural network的通用网络结构

[#fig:GNN_common_architecture]

总体说，

![GNN通用网络结构](figs/illustration/GNN_common_architecture.png){#fig:GNN_common_architecture width=60%}


[#fig:GNN_Unit]


![GNN单元](figs/illustration/GNN_Unit.png){#fig:GNN_Unit width=60%}


## 2.2 图神经网络的分类

[@tbl:gnn_overview]中列出了我们调研到的典型的图神经网络算法.表中列出了各个GNN中点/边计算的表达式,表达式中的大写粗体字母表示GNN模型参数.表中的网络类型划分依据[@zhou2018_gnn_review].因为本文主要关注GNN算法的计算特性,我们分析了各GNN算法的点、边计算的计算复杂度,并根据计算复杂度将GNN算法划分到四个象限中,如[@fig:GNN_complexity_quadrant]所示.

|          名称          |            网络类型             | 边计算 $\Sigma$  | 边计算 $\phi$                                                                                                                                                                                                                                                                                                                                                                                                                              |          边计算复杂度           | 点计算 $\gamma$                                                                                                                                                                                                                                                                                                                                                                                                                                                         |                     点计算复杂度                     |
| :--------------------: | :-----------------------------: | :--------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------: |
|  ChebNet (ICLR, 2016)  |        Spectral Methods         | sum              | $\vec{m}_{ij, k}^l = T_k(\widetilde{L} )_{ij} \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                 |         $O(K * h_{in})$         | $\vec{h}_i^{l+1} = \sum_{k=0}^K \mathbf{W}^k \cdot \vec{s}_{i, k}^{l} $                                                                                                                                                                                                                                                                                                                                                                                                 |                $O(h_{in} * h_{out})$                 |
|    GCN (ICLR, 2017)    |        Spectral Methods         | sum              | $\vec{m}_{ij}^l = e_{ij} \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                                      |           $O(h_{in})$           | $\vec{h}_i^{l+1} = \mathbf{W} \cdot \vec{s}_i^{l}$                                                                                                                                                                                                                                                                                                                                                                                                                      |                $O(h_{in} * h_{out})$                 |
|   AGCN (AAAI, 2018)    |        Spectral Methods         | sum              | $\vec{m}_{ij}^l = \tilde{e}_{ij}^l \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                            |           $O(h_{in})$           | $\vec{h}_i^{l+1} = \mathbf{W} \cdot \vec{s}_i^{l}$                                                                                                                                                                                                                                                                                                                                                                                                                      |                $O(h_{in} * h_{out})$                 |
| GraphSAGE(NIPS, 2017)  |          Non-spectral           | sum, mean, max   | $\vec{m}_{ij}^l =  \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                                            |             $O(1)$              | $\vec{h}_i^{l+1} =   \delta(\mathbf{W} \cdot [\vec{s}_i^{l} \parallel \vec{h}_i^l])$                                                                                                                                                                                                                                                                                                                                                                                    |                $O(h_{in} * h_{out})$                 |
| Neural FPs(NIPS, 2015) |      Non-spectral Methods       | sum              | $\vec{m}_{ij}^l = \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                                             |           $O(h_{in})$           | $\vec{h}_i^{l+1} = \delta(\mathbf{W}^{\mathbf{N}_i} \cdot \vec{s}_i^{l})$                                                                                                                                                                                                                                                                                                                                                                                               |                $O(h_{in} * h_{out})$                 |
|    SSE(ICML, 2018)     | Recurrent Graph Neural Networks | sum              | $\vec{m}_{ij}^l = [\vec{h}_i^{l} \parallel \vec{h}_j^l]$                                                                                                                                                                                                                                                                                                                                                                                   |             $O(1)$              | $\vec{h}_i^{l+1} = (1 - \alpha) \cdot \vec{h}_i^l +\alpha   \cdot \delta(\mathbf{W}_1 \delta(\mathbf{W}_2), \vec{s}_i^l)$                                                                                                                                                                                                                                                                                                                                               |                $O(h_{in} * h_{out})$                 |
|    GGNN(ICLR, 2015)    |   Gated Graph Neural Networks   | sum              | $m_{ij}^l = \mathbf{W} \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                                        |      $O(h_{in} * h_{out})$      | $\vec{z}_i^l = \delta ( \mathbf{W}^z \vec{s}_i^l + \mathbf{b}^{sz} + \mathbf{U}^z \vec{h}_i^{l} + \mathbf{b}^{hz}) \\ \vec{r}_i^l = \delta ( \mathbf{W}^r \vec{s}_i^l+ \mathbf{b}^{sr} +\mathbf{U}^r \vec{h}_i^{l} + \mathbf{b}^{hr}) \\ \vec{h}_i^{l+1} = tanh ( \mathbf{W} \vec{s}_i^l + \mathbf{b}^s + \mathbf{U} ( \vec{r}_i^l \odot \vec{h}_i^{l} + \mathbf{b}^h))) \\ \vec{h}_i^{l+1} = (1 - \vec{z}_i^l) \odot \vec{h}_i^l +  \vec{z}_i^l \odot \vec{h}_i^{l+1}$ |         $O(max(h_{in}, h_{out}) * h_{out})$          | $O(h_{in}, h_{out})$ |
|  Tree-LSTM(ACL, 2015)  |           Graph LSTM            | sum              | $\vec{m}_{ij}^l = \vec{h}_j^l$                                                                                                                                                                                                                                                                                                                                                                                                             |             $O(1)$              | $h_i^{l+1} = LSTM(\vec{s}_i^l, \vec{h}_i^{l})$                                                                                                                                                                                                                                                                                                                                                                                                                          |                $O(h_{in} * h_{out})$                 |
|    GAT(ICLR, 2017)     |    Graph Attention Networks     | sum, mean        | $\alpha_{ij}^k = \frac {\exp(LeakyReLU(a^T [ \mathbf{W}^k \cdot \vec{h}_i^l \parallel \mathbf{W}^k \cdot \vec{h}_j^l] ))} {\sum_{k \in \mathcal{N}(i)}\exp(LeakyReLU(a^T [ \mathbf{W}^k \cdot \vec{h}_i^l \parallel \mathbf{W}^k \cdot \vec{h}_k^l] ))} \\  \vec{m}_{ij}^l = \parallel_{k=1}^K \delta(\alpha_{ij}^k \mathbf{W}^k \vec{h}_j^{l})$                                                                                           |    $O(K * h_{in} * h_{out})$    | $\vec{h}_i^{l+1} = \vec{s}_i^l$                                                                                                                                                                                                                                                                                                                                                                                                                                         |                        $O(1)$                        |
|    GaAN(UAI, 2018)     |    Graph Attention Networks     | sum + max + mean | $\alpha_{ij}^k = \frac {\exp(\mathbf{W}^a \cdot [ \mathbf{W}^a \cdot \vec{h}_i^l \parallel \mathbf{W}^a \cdot \vec{h}_j^l] )} {\sum_{k \in \mathcal{N}(i)}\exp(a^T [ \mathbf{W}^k \cdot \vec{h}_i^l \parallel \mathbf{W}^k \cdot \vec{h}_k^l] )} \\  \vec{m}_{ij, 1}^l = \parallel_{k=1}^K \delta(\alpha_{ij}^k \mathbf{W}^k_v \vec{h}_j^{l}) \\ \vec{m}_{ij, 2}^l = \mathbf{W}_m \cdot \vec{h}_j^{l} \\ \vec{m}_{ij, 3}^l = \vec{h}_j^l $ | $O(max(d_a, d_m) * K * h_{in})$ | $\vec{g}_i = \mathbf{W}_g \cdot [\vec{h}_i^{l} \parallel s_{i, 2}^l \parallel s_{i, 3}^l]  \\ \vec{h}_i^{l+1} = \mathbf{W}_o [\vec{h}_i^l \parallel (\vec{g}_{i} \odot s_{i, 3}^l) ]$                                                                                                                                                                                                                                                                                   | $O(max(h_{in} + K * d_v, 2 * h_{in} + d_m) h_{out})$ |

: 图神经网络概览 {#tbl:gnn_overview}

![GNN的计算复杂度象限图](figs/illustration/GNN_complexity_quadrant.jpg){#fig:GNN_complexity_quadrant width=60%}

## 2.3 典型图神经网络

1. GCN

2. GGNN

3. GAT

4. GaAN

## 2.4 采样技术

根据对采样技术的调研，我们

## 2.5 图神经网络训练中的梯度更新


# 3 实验设计

## 3.1 实验环境

## 3.2 实验数据集

|                       数据集                        |  点数   |  边数   | 平均度数 | 输入特征向量维度 | 特征稀疏度 | 类别数 | 图类型 |
| :-------------------------------------------------: | :-----: | :-----: | :------: | :--------------: | :--------: | :----: | :----: |
| pubmed (pub) [@yang2016_revisiting_semisupervised]  | 19,717  | 44,324  |   4.5    |       500        |    0.90    |   3    | 有向图 |
|   amazon-photo (amp) [@shchur2018_pitfall_of_gnn]   |  7,650  | 119,081 |   31.1   |       745        |    0.65    |   8    | 有向图 |
| amazon-computers (amc) [@shchur2018_pitfall_of_gnn] | 13,752  | 245,861 |   35.8   |       767        |    0.65    |   10   | 有向图 |
| coauthor-physics (cph) [@shchur2018_pitfall_of_gnn] | 34,493  | 247,962 |   14.4   |       8415       |   0.996    |   5    | 有向图 |
|         flickr (fli) [@zeng2020_graphsaint]         | 89,250  | 899,756 |   10.1   |       500        |    0.54    |   7    | 无向图 |
|        com-amazon (cam) [@yang2012_defining]        | 334,863 | 925,872 |   2.8    |        32        |    0.0     |   10   | 无向图 |

: 实验数据集概览 {#tbl:dataset_overview}

实验中为了测量图的关键拓扑特征(例如平均度数)对性能的影响情况, 我们也利用R-MAT生成器[@rmat-generator]生成随机图.
如果不额外说明, 随机图顶点的特征向量为随机生成的32维稠密向量, 将顶点随机分到10个类别中, 75%的顶点参与训练.

## 3.3 图神经网络算法选择与实现

## 3.4 数据处理方法

## 3.5 实验方案概览

- 实验 1：第2.2节中的计算复杂度分析是否与实际表现相符合？

# 4 实验结果与分析

## 4.1 实验1：超参数的影响

本实验的目标是通过观察GNN的超参数(例如$h_{in}$、$h_{out}$、$K$等)对训练耗时、显存使用的影响, 验证[@tbl:gnn_overview]中复杂度分析的准确性.

[@fig:exp_absolute_training_time]中比较了各GNN每个epoch的训练耗时,其排名为GaAN >> GAT > GGNN > GCN. 其耗时排名与复杂度分析相符. 因为图中边的数量一般远超点的数量, 因此边计算复杂度更高的GAT算法比点计算复杂度高的算法GGNN更耗时. [@fig:exp_absolute_training_time] 同时表明个别epoch的训练耗时异常地高, 其主要是由profiling overhead和python解释器的GC停顿造成.该现象证实了去处异常epoch的必要性.

<div id="fig:exp_absolute_training_time">

![pubmed](./figs/experiments/exp_absolute_training_time_comparison_pubmed.png){}
![amazon-photo](./figs/experiments/exp_absolute_training_time_comparison_amazon-photo.png){}
![amazon-computers](./figs/experiments/exp_absolute_training_time_comparison_amazon-computers.png){}

![coauthor-physics](./figs/experiments/exp_absolute_training_time_comparison_coauthor-physics.png){}
![flickr](./figs/experiments/exp_absolute_training_time_comparison_flickr.png){}
![com-amazon](./figs/experiments/exp_absolute_training_time_comparison_com-amazon.png){}

训练耗时的影响
</div>

根据[@tbl:gnn_overview]中的复杂度分析, 各GNN的点、边计算复杂度与各算法超参数(例如$h_{dim}$、$K$等)呈线性关系.
为了验证该线性关系, 我们测量了各GNN的训练时间随超参数的变化情况.

GCN和GGNN的计算复杂度受隐向量维度$h_{dim}$影响.
$h_{dim}$同时影响Layer0的输出隐向量维度和Layer1的输入隐向量维度（即$h_{dim}=h^0_{out}=h^1_{in})$.
[@fig:exp_hyperparameter_on_vertex_edge_phase_time_gcn]和[@fig:exp_hyperparameter_on_vertex_edge_phase_time_ggnn]展示了GCN和GGNN训练耗时受$h_{dim}$的影响情况.
随着$h_{dim}$的增加,训练耗时呈线性增长.


GAT采用了多头机制,其计算复杂度受输入隐向量维度$h_{in}$, 每个头的隐向量维度$h_{head}$和头数$K$的影响.
每一层的输出隐向量维度$h_{out}=K h_{head}$.
因为在GAT结构中$h^1_{in}=h^0_{out}$, 调整$h_{head}$和$K$即相当于调整了Layer1的$h^1_{in}$.
[@fig:exp_hyperparameter_on_vertex_edge_phase_time_gat]展示了GAT训练耗时受超参数$h_{head}$和$K$的影响.
GAT训练耗时随$h_{head}$和$K$呈线性增长.

GaAN同样采用多头机制,其计算复杂度受$h_{in}$、$d_v$、$d_a$和头数$K$的影响.
[@fig:exp_hyperparameter_on_vertex_edge_phase_time_gat]展示了GaAN训练耗时受超参数的影响.
实验验证了[@tbl:gnn_overview]中给出的复杂度分析结果,各GNN算法的训练耗时随着超参数的增加呈线性增长.
当隐向量维度$h_{in}$过低时, 涉及隐向量的计算占总计算时间比例很低, 导致其总训练耗时变化不明显.
当隐向量维度足够大时, 总训练时间随$h_{in}$呈线性增长.

<div id="fig:exp_hyperparameter_on_vertex_edge_phase_time">

![GCN](figs/experiments/exp_hyperparameter_on_vertex_edge_phase_time_gcn.png){#fig:exp_hyperparameter_on_vertex_edge_phase_time_gcn}
![GGNN](figs/experiments/exp_hyperparameter_on_vertex_edge_phase_time_ggnn.png){#fig:exp_hyperparameter_on_vertex_edge_phase_time_ggnn}

![GAT](figs/experiments/exp_hyperparameter_on_vertex_edge_phase_time_gat.png){#fig:exp_hyperparameter_on_vertex_edge_phase_time_gat}
![GaAN](figs/experiments/exp_hyperparameter_on_vertex_edge_phase_time_gaan.png){#fig:exp_hyperparameter_on_vertex_edge_phase_time_gaan}

超参数对GNN中点/边计算耗时的影响
</div>

[@fig:exp_hyperparameter_on_memory_usage]同时展示了各GNN对GPU显存的使用情况随算法超参数的变化情况.
随着超参数的增加,GNN的显存使用也线性增长.

<div id="fig:exp_hyperparameter_on_memory_usage">


![GCN](figs/experiments/exp_hyperparameter_on_memory_usage_gcn.png){}
![GGNN](figs/experiments/exp_hyperparameter_on_memory_usage_ggnn.png){}

![GAT](figs/experiments/exp_hyperparameter_on_memory_usage_gat.png){}
![GaAN](figs/experiments/exp_hyperparameter_on_memory_usage_gaan.png){}

超参数对训练阶段显存使用的影响(不含数据集本身)

</div>

实验验证了[@tbl:gnn_overview]中复杂度分析的有效性.
*GNN的训练耗时与显存使用均与超参数呈线性关系*.
这允许算法工程师使用更大的超参数来提升GNN的复杂度,而不用担心训练耗时和显存使用呈现爆炸性增长.

## 4.2 实验2: 训练耗时分解

本实验的目标是通过对训练耗时的分解, 发掘GNN训练中的计算性能瓶颈.

对于点计算和边计算, [@fig:exp_vertex_edge_cal_proportion]展示了各算法不同GNN层点/边计算耗时占总训练耗时的比例情况(含forward, backward和evaluation阶段).
GCN算法在大多数数据集上边计算耗时占据主导.
只有`cph`数据集是特例, 因为该数据集输入特征向量维度非常高, 导致Layer0的点计算耗时额外的高.
GGNN因为其点计算复杂度高, 使其点计算耗时占比明显高于其他算法, 但在大多数数据集上依然是边计算占据主要的计算耗时.
只有在`pub`和`cam`数据集上,边计算开销和点计算开销接近,因为两个数据集平均度数较低 (仅为4.5和2.8).
对于GAT和GaAN算法, 因为其边计算复杂度高, 其边计算耗时占绝对主导.
综上, *边计算是GNN训练的主要耗时因素*, 尤其是在边计算较为复杂的情况下.

实验也表明*数据集的平均度数影响点/边计算的耗时比例*.
我们固定图的顶点数为50k, 利用R-MAT生成器生成平均度数在10到100之间的随机图.
我们测量了各GNN中点/边计算的耗时比例随图平均度数的变化情况, 如[@fig:exp_avg_degree_vertex_edge_cal_time]所示.
边计算的耗时随着平均度数的增加呈线性增长, *边计算耗时在绝大部分情况下主导了整个计算耗时*, 只有在点计算复杂度非常高且平均度数非常低的情况下点计算耗时才能赶超边计算耗时.
因此, *GNN训练优化的重点应该是提升边计算的效率*.

<div id="fig:exp_vertex_edge_cal_proportion">
![GCN](./figs/experiments/exp_layer_time_proportion_gcn.png)
![GGNN](./figs/experiments/exp_layer_time_proportion_ggnn.png)

![GAT](figs/experiments/exp_layer_time_proportion_gat.png)
![GaAN](figs/experiments/exp_layer_time_proportion_gaan.png)

点/边计算耗时占比
</div>

<div id="fig:exp_avg_degree_vertex_edge_cal_time">
![GCN](figs/experiments/exp_avg_degree_on_vertex_edge_cal_time_gcn.png)
![GGNN](figs/experiments/exp_avg_degree_on_vertex_edge_cal_time_ggnn.png)

![GAT](figs/experiments/exp_avg_degree_on_vertex_edge_cal_time_gat.png)
![GaAN](figs/experiments/exp_avg_degree_on_vertex_edge_cal_time_gaan.png)

平均顶点度数对点/边计算耗时比例的影响
</div>

边计算阶段可以进一步分解为collect, message, aggregate和update四个步骤, 如[@fig:steps_in_edge_calculation]所示.

![边计算的步骤](figs/illustration/steps_in_edge_calculation.png){#fig:steps_in_edge_calculation}

## 4.2 GPU显存使用

## 4.3 算法超参数影响

## 4.4 数据扩展性

## 4.5 采样技术的影响

# 5 系统设计建议

# 6 相关工作

# 7 总结与展望
