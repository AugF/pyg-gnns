#!/usr/bin/env bash
dir_config="dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gcn ggnn gat gaan)
layers=(2 3 4 5 6 7)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for l in ${layers[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, layers=${l}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${l}.json" ]; then # 断点续传
                continue
            fi
            python ../main.py --dataset ${data} --model ${model} --layers $l --json_path "${dir_config}/config0_${model}_${data}_${l}.json"
        done
    done
done
