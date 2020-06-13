#!/usr/bin/env bash
base_path=$(cd `dirname $0`; pwd)
dir_config="${base_path}/dir_qdrep"
dir_sqlite="${base_path}/dir_sqlite"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
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
            qdrep_path="${dir_config}/config0_${model}_${data}_${var}_20"
            if [ -f "${qdrep_file}.qdrep" ]; then # 断点续传
                continue
            fi
            val="configuration=0, model=${model}, dataset=${data}, var=${var}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_${var}_20" -w true python "${base_path}/../main.py" --dataset "${data}_${var}_20" --model ${model}
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_${var}_20.qdrep" "${dir_sqlite}/config0_${model}_${data}_${var}_20.sqlite"
        done
    done
done
