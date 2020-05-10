## 数据集格式
图拓扑，稀疏矩阵，行：顶点，列：顶点
顶点特征向量，稀疏/稠密矩阵，行：顶点，列：特征
顶点label，稠密矩阵，行：顶点，列（1维）：one-hot编码
稀疏矩阵采用scipy的稀疏矩阵存储格式。
稠密矩阵采用numpy的稠密矩阵存储格式。
统一用pickle.dump方法存储为了二进制文件

1. id-map.txt:  new_id到old_id的映射说明，之后的文件都已经将old_id替换为了new_id
2. ind.dataset.graph: 数据格式为python collections中defaultdict，存储的邻接表（实际处理中加了set），用pickle的dump方法保存为二进制文件
3. ind.dataset.allx: 数据格式为ndarray (稠密) 或scipy sparse的csr_matrix(稀疏）， 用pickle的dump方法保为二进制文件  
4. ind.dataset.ally: 数据格式为ndarray，保存为了one-hot编码，用pickle的dump方法保存为了二进制文件

## 预处理 
com-amazon和com-lj处理方法
1. 所有顶点按原始编号从小到大排序，去重，然后再统一重新编号，得到新的id， 保存在id-map文件中（每行(分隔符为\t, 从1开始标号)：new_id  old_id）
2. 原本cmty.txt中每一行表示一个社区，即一类，现在要映射到10类，采样的是每行对10取模，得到了各节点的类别
3. 特征获取方法：numpy的random.randint(0, 2)给每个nodes的每一维赋值，随机种子为1.

## 代码
ppi_preprocess.py: 预处理ppi数据集
reddit_preprocess.py: 预处理ppi数据集
preprocess.cpp: 对没有编号的顶点进行编号，得到id-map.txt文件
preprocess-graph.py, preprocess-xy.py: 预处理com-amazon, com-lj