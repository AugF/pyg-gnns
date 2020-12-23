#!/usr/bin/env bash
dir_config="dir_gat_inference_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gat)
hds=(8 16 32 64 128 256)
heads=(1 2 4 8 16)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for hd in ${hds[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, head_dims=${hd}, heads=4"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_4_${hd}.json" ]; then # 断点续传
                continue
            fi
            python ../main_inference_memory.py --device cuda:0 --dataset ${data} --model ${model} --head_dims ${hd} --d_v ${hd} --d_a ${hd} --heads 4 --json_path "${dir_config}/config0_${model}_${data}_4_${hd}.json"
        done
        for h in ${heads[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, head_dims=32, heads=${h}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${h}_32.json" ]; then # 断点续传
                continue
            fi
            python ../main_inference_memory.py --device cuda:0 --dataset ${data} --model ${model} --head_dims 32 --d_v 32 --d_a 32 --heads ${h} --json_path "${dir_config}/config0_${model}_${data}_${h}_32.json"
        done
    done
done

