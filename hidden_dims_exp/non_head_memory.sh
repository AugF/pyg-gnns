#!/usr/bin/env bash
dir_config="dir_json"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi


datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(ggnn)
hds=(2048)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for hd in ${hds[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, hidden_dims=${hd}"
            echo ${val}
            python ../main.py --dataset ${data} --model ${model} --json_path "${dir_config}/config0_${model}_${data}_${hd}.json" --epochs 5 --hidden_dims ${hd}
        done
    done
done

