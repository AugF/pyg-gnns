base_path=$(cd `dirname $0`; pwd)
git commit -am "add ..."
git checkout memory
date
bash "${base_path}/dense_feats_memory.sh" >>"${base_path}/dense_feats_memory.log" 2>&1
date
echo "finsh dense_feats_memory"

git commit -am "add ..."
git checkout master
bash "${base_path}/dense_feats_exp.sh" >>"${base_path}/dense_feats_exp.log" 2>&1
date
echo "finsh dense_feats_exp"