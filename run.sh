dir_path='/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns'

# date
# cd "${dir_path}/sampling_exp"
# bash run_graph_memory.sh >>run_graph_memory.log 2>&1
# echo "graph_memory_exp end"

# 1. config exp
date
echo "begin config_exp"
cd "${dir_path}/config_exp"
bash config_exp.sh >>config_exp.log 2>&1
echo "exp end ..."
date
bash config_memory.sh >>config_memory.log 2>&1
echo "memory end ..."

# 2. hidden_dims_exp
date
echo "begin hidden_dims_exp"
cd "${dir_path}/hidden_dims_exp"
bash multi_head_exp.sh >>multi_head_exp.log 2>&1
echo "multi exp end ..."
date
bash multi_head_memory.sh >>multi_head_memory.log 2>&1
echo "multi memory end ..."
date
bash non_head_exp.sh >>non_head_exp.log 2>&1
echo "non exp end ..."
date
bash non_head_memory.sh >>non_head_memory.log 2>&1
echo "non memory end ..."

# 3. hidden_dims_3_exp
# date
# echo "begin hidden_dims_3_exp"
# cd "${dir_path}/hidden_dims_3_exp"
# bash multi_head_exp.sh >>multi_head_exp.log 2>&1
# echo "multi exp end ..."
# date
# bash multi_head_memory.sh >>multi_head_memory.log 2>&1
# echo "multi memory end ..."
# date
# bash non_head_exp.sh >>non_head_exp.log 2>&1
# echo "non exp end ..."
# date
# bash non_head_memory.sh >>non_head_memory.log 2>&1
# echo "non memory end ..."

# 4. layer_exp
date
echo "begin layer_exp"
cd "${dir_path}/layer_exp"
bash layer_exp.sh >>layer_exp.log 2>&1
echo "exp end..."
date
bash layer_memory.sh >>layer_memory.log 2>&1
echo "memory end..."

# 5. dense_feats_exp
date
echo "begin dense_feats..."
cd "${dir_path}/dense_feats_exp"
bash dense_feats_exp.sh >>dense_feats_exp.log 2>&1
echo "exp end..."
date
bash dense_feats_memory.sh >>dense_feats_memory.log 2>&1
echo "memory end..."
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