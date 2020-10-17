#!/usr/bin/env bash
dir_config="batch_train_time_stack"

datasets=(amazon-computers flickr)
models=(gaan)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        file_path="${dir_config}/config0_${model}_${data}_full.log"
        if [ -f $file_path ]; then # 断点续传
            continue
        fi
        echo $file_path
        date
        python ../main_full.py --dataset ${data} --model ${model} --epochs 50 >>${file_path} 2>&1
    done
done

