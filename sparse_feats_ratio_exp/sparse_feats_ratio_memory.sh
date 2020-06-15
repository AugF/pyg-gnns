#!/usr/bin/env bash
base_path=$(cd `dirname $0`; pwd)
dir_config="${base_path}/dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gcn ggnn gat gaan)
vars=(5 10 20 50)

for model in ${models[@]}
do
    for data in ${datasets[@]}
    do
        for var in ${vars[@]}
        do
            json_path="${dir_config}/config0_${model}_${data}_500_${var}.json"
            if [ -f $json_path ]; then # 断点续传
                continue
            fi
            val="configuration=0, model=${model}, dataset=${data}, var=${var}"
            echo ${val}
            python -u "${base_path}/../main.py" --dataset "${data}_500_${var}" --model ${model} --json_path "${dir_config}/config0_${model}_${data}_500_${var}.json"
        done
    done
done
