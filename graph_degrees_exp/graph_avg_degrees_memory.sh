#!/usr/bin/env bash
dir_config="dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

models=(gcn ggnn gat gaan)
degrees=(3 6 10 15 20 25 30 50)

for model in ${models[@]}
do
    for ds in ${degrees[@]}
    do
        json_path="${dir_config}/config0_${model}_graph_50k_${ds}.json"
        if [ -f $json_path ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_50k_${ds}"
        echo ${val}
        python ../main.py --dataset "graph_50k_${ds}" --model ${model} --json_path "${dir_config}/config0_${model}_graph_50k_${ds}.json"
    done
done
