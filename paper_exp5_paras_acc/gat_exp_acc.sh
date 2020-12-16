#!/usr/bin/env bash
dir_config="dir_gat_acc"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr)
models=(gat)
hds=(1 2 4 8 16 32 64 128 256)
heads=(1 2 4 8 16)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        # for hd in ${hds[@]}
        # do
        #     val="model=${model}, dataset=${data}, head_dims=${hd}, heads=4"
        #     echo ${val}
        #     if [ -f "${dir_config}/config0_${model}_${data}_4_${hd}.log" ]; then # 断点续传
        #         continue
        #     fi
        #     python ../main_kdd_criterion.py --device cuda:1 --dataset ${data} --model ${model} --head_dims ${hd} --d_v ${hd} --d_a ${hd} --heads 4 1>"${dir_config}/config0_${model}_${data}_4_${hd}.log" 2>&1
        # done
        for h in ${heads[@]}
        do
            val="model=${model}, dataset=${data}, head_dims=32, heads=${h}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${h}_32.log" ]; then # 断点续传
                continue
            fi
            python ../main_kdd_criterion.py --device cuda:1 --dataset ${data} --model ${model} --head_dims 32 --heads ${h} 1>"${dir_config}/config0_${model}_${data}_${h}_32.log" 2>&1
        done
    done
done

