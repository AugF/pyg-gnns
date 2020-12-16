#!/usr/bin/env bash
dir_config="dir_gat_heads_exp"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr)
model="gat"
hds=(1 2 4 8)
heads=(1 2 4 8 16)

for data in ${datasets[@]}
do
    for hd in ${hds[@]}
    do
        for h in ${heads[@]}
        do
            val="model=${model}, dataset=${data}, head_dims=${hd}, heads=${h}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${h}_${hd}.log" ]; then # 断点续传
                continue
            fi
            python ../main_kdd_criterion.py --device cuda:1 --dataset ${data} --model ${model} --head_dims ${hd} --heads ${h} 1>"${dir_config}/config0_${model}_${data}_${h}_${hd}.log" 2>&1
        done
    done
done

