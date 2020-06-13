base_path=$(cd `dirname $0`; pwd)
date
bash "${base_path}/dense_feats_memory.sh" >>"${base_path}/dense_feats_memory.log" 2>&1
date
echo "finsh dense_feats_memory"

bash "${base_path}/dense_feats_exp.sh" >>"${base_path}/dense_feats_exp.log" 2>&1
date
echo "finsh dense_feats_exp"