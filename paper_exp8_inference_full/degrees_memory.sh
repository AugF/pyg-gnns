#!/usr/bin/env bash
dir_config="dir_degrees_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

models=(gcn ggnn gat gaan)
degrees=(2 5 10 15 20 30 40 50 70)

for model in ${models[@]}
do
    for ds in ${degrees[@]}
    do
        json_path="${dir_config}/config0_${model}_graph_10k_${ds}.json"
        if [ -f $json_path ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_10k_${ds}"
        echo ${val}
        python ../main_inference_memory.py --device cuda:0 --dataset "graph_10k_${ds}" --model ${model} --json_path "${dir_config}/config0_${model}_graph_10k_${ds}.json"
    done
done
