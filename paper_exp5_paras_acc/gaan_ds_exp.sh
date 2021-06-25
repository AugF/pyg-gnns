#!/usr/bin/env bash
dir_config="dir_gaan_ds_acc"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr)
models=(gaan)
ds=(1 2 4 8 16 32 64 128 256)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for hd in ${ds[@]}
        do
            val="model=${model}, dataset=${data}, heads=1, d_a, d_v, d_m=${hd}, hidden_dims=8"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_1_${hd}_8.log" ]; then # 断点续传
                continue
            fi
            python ../main_kdd_criterion.py --device cuda:0 --dataset ${data} --model ${model} --d_v ${hd} --d_a ${hd} --d_m ${hd} --heads 1 --hidden_dims 8 1>"${dir_config}/config0_${model}_${data}_1_${hd}_8.log" 2>&1
        done
    done
done
