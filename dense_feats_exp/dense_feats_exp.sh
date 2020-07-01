#!/usr/bin/env bash
dir_config="dir_qdrep"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
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
            nsys profile -t cuda,osrt,nvtx -f true --export=sqlite -o "${dir_config}/config0_${model}_${data}_${fd}" -w true python ../main.py --dataset "${data}_${fd}" --model ${model}
        done
    done
done
