#!/usr/bin/env bash
base_path=$(cd `dirname $0`; pwd)
dir_config="${base_path}/dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

models=(gcn ggnn gat gaan)
nodes=(1 25 50 75 100)

for model in ${models[@]}
do
    for ns in ${nodes[@]}
    do
        json_path="${dir_config}/config0_${model}_graph_${ns}k_25.json"
        if [ -f $json_path ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_${ns}k_25"
        echo ${val}
        python "${base_path}/../main.py" --dataset "graph_${ns}k_25" --model ${model} --json_path "${dir_config}/config0_${model}_graph_${ns}k_25.json"
    done
done
