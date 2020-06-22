#!/usr/bin/env bash
dir_config="${base_path}/dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gcn ggnn gat gaan)
vars=(250 500 750 1000 1250)

for model in ${models[@]}
do
    for data in ${datasets[@]}
    do
        for var in ${vars[@]}
        do
            json_path="${dir_config}/config0_${model}_${data}_${var}_20.json"
            if [ -f $json_path ]; then # 断点续传
                continue
            fi
            val="configuration=0, model=${model}, dataset=${data}, var=${var}"
            echo ${val}
            python -u ../main.py --dataset "${data}_${var}_20" --model ${model} --json_path ${json_path}
        done
    done
done
