#!/usr/bin/env bash
dir_config="batch_memory"

datasets=(amazon-computers flickr)
models=(gcn ggnn gat gaan)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        file_path="${dir_config}/config0_${model}_${data}_full.json"
        if [ -f $file_path ]; then # 断点续传
            continue
        fi
        echo $file_path
        date
        python ../main_full.py --dataset ${data} --model ${model} --epochs 20 --json_path ${file_path}
    done
done

