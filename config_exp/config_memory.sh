#!/usr/bin/env bash
# experiment for configuration
dir_config="dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gcn ggnn gat gaan)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        val="configuration=${i}, model=${model}, dataset=${data}"
        echo ${val}
        python ../main.py --dataset ${data} --model ${model} --json_path "${dir_config}/config0_${model}_${data}.json"
    done
done
