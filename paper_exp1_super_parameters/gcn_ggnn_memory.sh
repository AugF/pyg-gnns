#!/usr/bin/env bash
dir_config="dir_gcn_ggnn_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gcn ggnn)
hds=(16 32 64 128 256 512 1024 2048)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for hd in ${hds[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, hidden_dims=${hd}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${hd}.json" ]; then # 断点续传
                continue
            fi
            python ../main_memory.py --device cuda:0 --dataset ${data} --model ${model} --json_path "${dir_config}/config0_${model}_${data}_${hd}.json" --hidden_dims ${hd}
        done
    done
done

