dir_path='/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns'

date
# 0. run_time_exp
echo "begin sampling_exp"
cd "${dir_path}/sampling_exp"
# bash run_json.sh >run_json.log 2>&1
# echo "run_json end ..."
# date
# bash run_nsys.sh >run_nsys.log 2>&1
# echo "run_nsys end ..."
# date
# bash run_time.sh >run_time.log 2>&1
# echo "run_time end ..."
date
echo "begin dense feature sampling exp"
bash run_dense_features_memory.sh >>run_dense_features_memory.log 2>&1
echo "dense memory end"
date
bash run_dense_features_time.sh >>run_dense_features_time.log 2>&1
echo "dense time end"

date
echo "begin sparse feature sampling exp"
bash run_sparse_features_memory.sh >>run_sparse_features_memory.log 2>&1
echo "sparse memory end"
date
bash run_sparse_features_time.sh >>run_sparse_features_time.log 2>&1
echo "sparse time end"

date
echo "begin graph sampling exp"
bash run_graph_memory.sh >>run_graph_memory.log 2>&1
echo "graph memory end"
date
bash run_graph_time.sh >>run_graph_time.log 2>&1
echo "graph time end"
date