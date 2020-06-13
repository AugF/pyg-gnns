#!/usr/bin/env bash
dir_config="dir_qdrep"
dir_sqlite="dir_sqlite"

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
layers=(2 3 4 5 6 7)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for l in ${layers[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, layers=${l}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_${l}" -w true python ../main.py --dataset ${data} --model ${model} --layers $l 
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_${l}.qdrep" "${dir_sqlite}/config0_${model}_${data}_${l}.sqlite"
        done
    done
done
