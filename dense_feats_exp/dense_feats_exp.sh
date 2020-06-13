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
fds=(16 32 64 128 256 512)

for model in ${models[@]}
do
    for data in ${datasets[@]}
    do
        for fd in ${fds[@]}
        do
            qdrep_path="${dir_config}/config0_${model}_${data}_${fd}"
            if [ -f "${qdrep_file}.qdrep" ]; then # 断点续传
                continue
            fi
            val="configuration=0, model=${model}, dataset=${data}, dense_feat_dims=${fd}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_${fd}" -w true python "${base_path}/../main.py" --dataset "${data}_${fd}" --model ${model}
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_${fd}.qdrep" "${dir_sqlite}/config0_${model}_${data}_${fd}.sqlite"
        done
    done
done
