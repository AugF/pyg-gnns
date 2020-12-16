export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
dir_path="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns"
date

# cd "${dir_path}/paper_exp5_paras_acc"
# echo "gat heads acc"
# bash gat_heads_exp.sh >gat_heads_exp.log 2>&1
# date

cd "${dir_path}/paper_exp6_sampling_acc"
echo "graphsage fix time"
bash sh_graphsage_fix_time_new.sh >sh_graphsage_fix_time_new.log 2>&1
date