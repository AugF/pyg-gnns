## 基于GPU的图神经网络计算性能瓶颈分析

pyg-analysis是pyg-gnns的脚本代码。

### 目录结构

```
cfg_file                        // 配置文件
paper_exp1_super_parameters     // 模型超参数对性能的影响
paper_exp2_time_break           // 时间耗时分解分析
paper_exp3_memory               // 内存使用分析
paper_exp4_relative_sampling    // 采样对性能的影响
paper_exp5_inference_full       // 全数据训练的推理阶段的分析
paper_exp5_inference_sampling   // 分批训练的推理阶段的分析
tools                           // 辅助目录
citation_datasets.py            // 标准文件
stages_exp.py                   // 前向传播、后向传播与评估时期分析脚本
calculation_exp.py              // 边/点计算时期分析脚本
layers_exp.py                   // 分层分析脚本
layers_calculations_exp.py      // 分层计算分析脚本
edge_cal_exp.py                 // 边计算层次分析脚本
operators_exp.py                // 基础算子耗时分析脚本
pygs_utils.py                   // 辅助文件
citation_datasets.py            // 数据集文件
datasets.py                     // 同上
pics_xxx: 绘图脚本
```

### 安装说明

这里统一采用pip安装方式：
1. 安装anaconda（[官方文档](https://docs.anaconda.com/anaconda/install/index.html)），创建新的环境`conda create -n optimize-pygs python==3.7.7`，并激活`conda activate optimize-pygs`
2. 安装`PyTorch1.5.0`, [官方文档](https://pytorch.org/), 执行命令`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
3. 安装`PyTorchGeomtric1.5.0`, [官方文档](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [PyG1.5.0+cu101](https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html)。
    ```
    pip install tools/torch_cluster-1.5.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_scatter-2.0.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_sparse-0.6.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_spline_conv-1.2.0-cp37-cp37m-linux_x86_64.whl
    pip install torch-geometric==1.5.0
    ```
4. 安装其他软件，`pip install -r requirements.txt`
5. 设置目录，将`neuroc_pygs/`文件中设置`dataset_root="xxx/mydata"`, `PROJECT_PATH="xxx/neuroc_pygs"`（即neuroc_pygs目录的绝对位置）。并执行`python setup.py install --user`保存修改。

### 安装说明

这里统一采用pip安装方式：
1. 安装anaconda（[官方文档](https://docs.anaconda.com/anaconda/install/index.html)），创建新的环境`conda create -n optimize-pygs python==3.7.7`，并激活`conda activate optimize-pygs`
2. 安装`PyTorch1.5.0`, [官方文档](https://pytorch.org/), 执行命令`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
3. 安装`PyTorchGeomtric1.5.0`, [官方文档](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [PyG1.5.0+cu101](https://pytorch-geometric.com/whl/torch-1.5.0+cu101.html)。
    ```
    pip install tools/torch_cluster-1.5.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_scatter-2.0.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_sparse-0.6.4-cp37-cp37m-linux_x86_64.whl
    pip install tools/torch_spline_conv-1.2.0-cp37-cp37m-linux_x86_64.whl
    pip install torch-geometric==1.5.0
    ```

### 运行环境

- 硬件环境
    - 2 × NVIDIA Tesla T4 GPU( 16GB)
    - CentOS 7 server, 40 cores, 90GB
- 软件环境：
    - Python3.7.7
    - PyTorch1.5.0
    - CUDA10.1
    - PyTorchGeometric1.5.0

### 运行说明

1. 激活环境`conda activate optimize-pygs`
2. 运行`python main.py`

### 算法信息

按照边/点计算复杂度划分GNN算法，并选取了典型算法。
1. 边低点低, [GCN](https://github.com/tkipf/gcn)
2. 边低点高, [GGNN](https://github.com/yujiali/ggnn)
3. 边高点低, [GAT](https://github.com/PetarV-/GAT)
4. 边高点高, [GaAN](https://github.com/jennyzhang0215/GaAN)

### 数据集信息

数据集位置: /mnt/data/wangzhaokang/wangyunpan/datasets

| 数据集 | 点数 | 边数 | 平均度数 | 特征数 | 类别数 | 有向图 |
| --- | --- | --- | --- | --- | --- | --- |
| pubmed | 19,717 | 44,324 | 4.5 | 500 | 3 | 是 |
| amazon-photo | 7,650 | 119,081 | 31.1 | 745 | 8 | 是 |
| amazon-computers | 13,752 | 245,861 | 35.8 | 767 | 10 | 是 |
| coauthor-physics | 34,493 | 247,962 | 14.4 | 8415 | 5 | 是 |
| flickr | 89,250 | 899,756 | 10.1 | 500 | 7 | 否 |
| com-amazon | 334,863 | 925,872 | 2.8 | 32 | 10 | 否 |
| reddit | 232,965 | 23,213,838 | 99.6 | 602 | 41 | 否 |
| yelp | 716,847 | 13,954,819 | 19.5 | 300 | 200 | 是 |
    
dataset格式(N为边数, F为特征数， C为类别数）
- adj_full.npz: a sparse matrix in csr format, ['indptr', 'indices', 'data', 'shape'], N * N
- feats.npy: a numpy array, N * F
- role.json: a dictionary of three keys. Key 'tr' corresponds to the list of all training node indices. Key va corresponds to the list of all validation node indices. Key te corresponds to the list of all test node indices. Note that in the raw data, nodes may have string-type ID. You would need to re-assign numerical ID (0 to N-1) to the nodes, so that you can index into the matrices of adj, features and class labels.
- class_map.json:a dictionary of length N. Each key is a node index, and each value is either a length C binary list (for multi-class classification) or an integer scalar (0 to C-1, for single-class classification)


