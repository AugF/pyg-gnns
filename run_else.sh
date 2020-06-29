dir_path='/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns'

# 5. sampling exp: data: sparse
date
echo "begin sampling exp: sparse"
cd "${dir_path}/sampling_exp/data_exp"
bash run_sparse_features_memory.sh >run_sparse_features_memory.log 2>&1
echo "sparse features memory end ..."
date
bash run_sparse_features_time.sh >run_sparse_features_time.log 2>&1
echo "sparse features time end ..."
date
bash run_graph_time.sh >run_graph_time.log 2>&1
echo "graph time end ..."
date

# 6. sparse_feats_dims_exp
echo "begin sparse_feats"
cd "${dir_path}/sparse_feats_dims_exp"
bash sparse_feats_dims_exp.sh >>sparse_feats_dims_exp.log 2>&1
echo "dims exp end ..."
date
bash sparse_feats_dims_memory.sh >>sparse_feats_dims_memory.log 2>&1
echo "dims memory end ..."
date

# 7. sparse_feats_ratio_exp
cd "${dir_path}/sparse_feats_ratio_exp"
bash sparse_feats_ratio_exp.sh >>sparse_feats_ratio_exp.log 2>&1
echo "ratio exp end ..."
date
bash sparse_feats_ratio_memory.sh >>sparse_feats_ratio_memory.log 2>&1
echo "ratio memory end ..."
date

# 8. graph_nodes_exp
echo "begin graph..."
cd "${dir_path}/graph_nodes_exp"
bash graph_nodes_exp.sh >>graph_nodes_exp.log 2>&1
echo "nodes exp end ..."
date
bash graph_nodes_memory.sh >>graph_nodes_memory.log 2>&1
echo "nodes memory end ..."
date

# 9. graph_degree_exp
cd "${dir_path}/graph_degrees_exp"
bash graph_degrees_exp.sh >>graph_degrees_exp.log 2>&1
echo "degrees exp end ..."
date
bash graph_degrees_memory.sh >>graph_degrees_memory.log 2>&1
echo "degrees memory end ..."
date

# 10. sparse x
cd "${dir_path}/sparse_x_exp"
bash sparse_feats_dims_exp.sh >>sparse_feats_dims_exp.log 2>&1
echo "dims exp end ..."
date
bash sparse_feats_ratio_exp.sh >>sparse_feats_ratio_exp.log 2>&1
echo "ratio exp end ..."
date