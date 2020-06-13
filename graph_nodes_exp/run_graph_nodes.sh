base_path=$(cd `dirname $0`; pwd)
date
bash "${base_path}/graph_nodes_memory.sh" >>"${base_path}/graph_nodes_memory.log" 2>&1
date
echo "finsh graph_nodes_memory"

bash "${base_path}/graph_nodes_exp.sh" >>"${base_path}/graph_nodes_exp.log" 2>&1
date
echo "finsh graph_nodes_exp"