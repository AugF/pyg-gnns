export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
dir_path="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns"

# date
# cd "${dir_path}/paper_exp2_time_break"
# echo "exp2: "
# bash config_exp.sh > config_exp.log 2>&1
# date
# bash degrees_exp.sh >degrees_exp.log 2>&1
# date
# bash config_memory.sh >config_memory.log 2>&1

date
cd "${dir_path}/paper_exp3_memory"
echo "exp3: "
bash feat_dims_memory.sh >feat_dims_memory.log 2>&1
# date
# bash degrees_memory.sh >degrees_memory.log 2>&1
date
bash fix_edges_memory.sh >fix_edges_memory.log 2>&1

