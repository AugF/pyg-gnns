#!/usr/bin/env bash
dir_config="dir_gaan_heads_acc"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr)
models=(gaan)
heads=(1 2 4 8 16)


for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        # heads
        for h in ${heads[@]}
        do
            val="model=${model}, dataset=${data}, heads=${h}, d_a, d_v, d_m=2, hidden_dims=4"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${h}_2_4.log" ]; then # 断点续传
                continue
            fi
            python ../main_kdd_criterion.py --device cuda:1 --dataset ${data} --model ${model} --d_m 2 --d_v 2 --d_a 2 --heads ${h} --hidden_dims 4 1>"${dir_config}/config0_${model}_${data}_${h}_2_4.log" 2>&1
        done
    done
done
