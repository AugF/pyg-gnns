# date
# echo "begin feats..."
# bash dense_feats_exp/run_dense_feats.sh
# bash sparse_feats_dims_exp/run_sparse_feats_dims.sh
# bash sparse_feats_ratio_exp/run_sparse_feats_ratio.sh

date
echo "begin graph..."
bash graph_nodes_exp/run_graph_nodes.sh
date
bash graph_degrees_exp/run_graph_degrees.sh
date