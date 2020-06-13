base_path=$(cd `dirname $0`; pwd)
git commit -am "add ..."
git checkout memory
date
bash "${base_path}/sparse_feats_dims_memory.sh" >>"${base_path}/sparse_feats_dims_memory.log" 2>&1
date
echo "finsh sparse_feats_dims_memory"

git commit -am "add ..."
git checkout master
bash "${base_path}/sparse_feats_dims_exp.sh" >>"${base_path}/sparse_feats_dims_exp.log" 2>&1
date
echo "finsh sparse_feats_dims_exp"