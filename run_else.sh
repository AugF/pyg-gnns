dir_path='/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns'

# 4. dense_feats_exp
date
echo "begin dense_feats..."
cd "${dir_path}/dense_feats_exp"
bash dense_feats_exp.sh 1>/dev/null 2>>dense_feats_exp.log
echo "exp end..."
date
bash dense_feats_memory.sh 1>/dev/null 2>>dense_feats_memory.log
echo "memory end..."
date

# 5. sampling exp: data: sparse
date
echo "begin sampling exp: sparse"
cd "${dir_path}/sampling_exp/data_exp"
bash run_sparse_features_memory.sh 1>/dev/null 2>run_sparse_features_memory.log
echo "sparse features memory end ..."
date
bash run_sparse_features_time.sh 1>/dev/null 2>run_sparse_features_time.log
echo "sparse features time end ..."
date
bash run_graph_time.sh 1>/dev/null 2>run_graph_time.log
echo "graph time end ..."
date

# 6. sparse_feats_dims_exp
echo "begin sparse_feats"
cd "${dir_path}/sparse_feats_dims_exp"
bash sparse_feats_dims_exp.sh 1>/dev/null 2>>sparse_feats_dims_exp.log
echo "dims exp end ..."
date
bash sparse_feats_dims_memory.sh 1>/dev/null 2>>sparse_feats_dims_memory.log
echo "dims memory end ..."
date

# 7. sparse_feats_ratio_exp
cd "${dir_path}/sparse_feats_ratio_exp"
bash sparse_feats_ratio_exp.sh 1>/dev/null 2>>sparse_feats_ratio_exp.log
echo "ratio exp end ..."
date
bash sparse_feats_ratio_memory.sh 1>/dev/null 2>>sparse_feats_ratio_memory.log
echo "ratio memory end ..."
date

# 8. graph_nodes_exp
echo "begin graph..."
cd "${dir_path}/graph_nodes_exp"
bash graph_nodes_exp.sh 1>/dev/null 2>>graph_nodes_exp.log
echo "nodes exp end ..."
date
bash graph_nodes_memory.sh 1>/dev/null 2>>graph_nodes_memory.log
echo "nodes memory end ..."
date

# 9. graph_degree_exp
cd "${dir_path}/graph_degrees_exp"
bash graph_degrees_exp.sh 1>/dev/null 2>>graph_degrees_exp.log
echo "degrees exp end ..."
date
bash graph_degrees_memory.sh 1>/dev/null 2>>graph_degrees_memory.log
echo "degrees memory end ..."
date

# 10. sparse x
cd "${dir_path}/sparse_x_exp"
bash sparse_feats_dims_exp.sh 1>/dev/null 2>>sparse_feats_dims_exp.log
echo "dims exp end ..."
date
bash sparse_feats_ratio_exp.sh 1>/dev/null 2>>sparse_feats_ratio_exp.log
echo "ratio exp end ..."
date