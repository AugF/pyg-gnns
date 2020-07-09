#!/usr/bin/env bash
dir_config="dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

models=(gcn ggnn gat gaan)
nodes=(1 5 10 20 30 40 50)

for model in ${models[@]}
do
    for ns in ${nodes[@]}
    do
        json_path="${dir_config}/config0_${model}_graph_${ns}k_20.json"
        if [ -f $json_path ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_${ns}k_20"
        echo ${val}
        python ../main.py --dataset "graph_${ns}k_20" --model ${model} --json_path "${dir_config}/config0_${model}_graph_${ns}k_20.json"
    done
done
