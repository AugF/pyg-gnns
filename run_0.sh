export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
dir_path="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns"

# date
# cd "${dir_path}/paper_exp7_inference_sampling"
# echo "exp7: "
# bash inference_sampling_cluster_time.sh >inference_sampling_cluster_time.log 2>&1
# date
# bash inference_sampling_cluster_memory.sh >inference_sampling_cluster_memory.log 2>&1

date
cd "${dir_path}/paper_exp6_sampling_acc"
echo "exp6: "
bash batch_acc_cum_cluster_fix_time.sh >batch_acc_cum_cluster_fix_time.log 2>&1

