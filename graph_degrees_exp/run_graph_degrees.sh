base_path=$(cd `dirname $0`; pwd)
git commit -am "add ..."
git checkout memory
date
bash "${base_path}/graph_degrees_memory.sh" >>"${base_path}/graph_degrees_memory.log" 2>&1
date
echo "finsh graph_degrees_memory"

git commit -am "add ..."
git checkout master
bash "${base_path}/graph_degrees_exp.sh" >>"${base_path}/graph_degrees_exp.log" 2>&1
date
echo "finsh graph_degrees_exp"