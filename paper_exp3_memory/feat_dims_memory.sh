#!/usr/bin/env bash
dir_config="dir_feat_dims_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


# datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
datasets=(com-amazon)
models=(gcn ggnn gat gaan)
fds=(16 32 64 128 256 512)

for model in ${models[@]}
do
    for data in ${datasets[@]}
    do
        for fd in ${fds[@]}
        do
            json_path="${dir_config}/config0_${model}_${data}_${fd}.json"
            if [ -f $json_path ]; then # 断点续传
                continue
            fi
            val="configuration=0, model=${model}, dataset=${data}, dense_feat_dims=${fd}"
            echo ${val}
            python -u ../main_memory.py --dataset "${data}_${fd}" --model ${model} --device cuda:1 --json_path "${dir_config}/config0_${model}_${data}_${fd}.json"
        done
    done
done
