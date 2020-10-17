dir_path="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns"

date
cd "${dir_path}/paper_exp1_super_parameters"
echo "exp1: "
# bash gaan_exp.sh > gaan_exp.log 2>&1
# date
bash gaan_memory.sh >gaan_memory.log 2>&1

date
cd "${dir_path}/paper_exp2_time_break"
echo "exp2: "
# bash config_exp.sh > config_exp.log 2>&1
# date
bash config_memory.sh >config_memory.log 2>&1
date
# bash degrees_exp.sh >degrees_exp.log 2>&1

date
cd "${dir_path}/paper_exp3_memory"
echo "exp3: "
bash feat_dims_memory.sh >feat_dims_memory.log 2>&1
date
bash degrees_memory.sh >degrees_memory.log 2>&1
date
bash fix_edges_memory.sh >fix_edges_memory.log 2>&1

date
cd "${dir_path}/paper_exp4_relative_sampling"
echo "exp4 "
bash sampling_batch_full_memory.sh >sampling_batch_full_memory.log 2>&1
date
bash sampling_batch_full_time.sh >sampling_batch_full_time.log 2>&1
date
python sampling_batch_memory.py
date
python sampling_batch_train_time_stack.py
date
