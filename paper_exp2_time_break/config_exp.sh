#!/usr/bin/env bash
# experiment for configuration
dir_config="dir_config_qdrep"
dir_sqlite="dir_config_sqlite"

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

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        val="configuration=0, model=${model}, dataset=${data}"
        echo ${val}
        if [ -f "${dir_config}/config0_${model}_${data}.qdrep" ]; then # 断点续传
            continue
        fi
        nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}" -w true python ../main_inference.py --dataset ${data} --model ${model}
        nsys-exporter -s "${dir_config}/config0_${model}_${data}.qdrep" "${dir_sqlite}/config0_${model}_${data}.sqlite"
    done
done

