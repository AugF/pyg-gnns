base_path=$(cd `dirname $0`; pwd)
date
bash "${base_path}/sparse_feats_ratio_memory.sh" >>"${base_path}/sparse_feats_ratio_memory.log" 2>&1
date
echo "finsh sparse_feats_ratio_memory"

bash "${base_path}/sparse_feats_ratio_exp.sh" >>"${base_path}/sparse_feats_ratio_exp.log" 2>&1
date
echo "finsh sparse_feats_ratio_exp"