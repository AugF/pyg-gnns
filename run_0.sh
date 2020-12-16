export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
dir_path="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns"
date

cd "${dir_path}/paper_exp7_inference_sampling"
echo "begin inference sampling memory"
bash inference_sampling_memory.sh > inference_sampling_memory.log 2>&1
date

cd "${dir_path}/paper_exp6_sampling_acc"
echo "cluster fix time"
bash sh_cluster_fix_time_new.sh >sh_cluster_fix_time_new.log 2>&1
date