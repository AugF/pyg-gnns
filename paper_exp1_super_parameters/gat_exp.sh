#!/usr/bin/env bash
dir_config="dir_gat_inference_qdrep"
dir_sqlite="dir_gat_inference_sqlite"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
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
            if [ -f "${dir_config}/config0_${model}_${data}_4_${hd}.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_4_${hd}" -w true python ../main_inference.py --dataset ${data} --model ${model} --head_dims ${hd} --d_v ${hd} --d_a ${hd} --heads 4
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_4_${hd}.qdrep" "${dir_sqlite}/config0_${model}_${data}_4_${hd}.sqlite"
        done
        for h in ${heads[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, head_dims=32, heads=${h}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${h}_32.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_${h}_32" -w true python ../main_inference.py --dataset ${data} --model ${model} --head_dims 32 --d_v 32 --d_a 32 --heads ${h}
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_${h}_32.qdrep" "${dir_sqlite}/config0_${model}_${data}_${h}_32.sqlite"
        done
    done
done
d
