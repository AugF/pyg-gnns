#!/usr/bin/env bash
# experiment for configuration
dir_config="config_exp"

if [ ! -d $dir_config ]
then
    mkdir $dir_config
fi

datasets=(flickr com-amazon reddit com-lj)
models=(gcn ggnn gat gaan)
hidden_dims=(64 16 8)
heads=(8 4 4)
head_dims=(8 4 2)
d_a=(8 4 2)
d_v=(8 4 2)

for i in 0 1 2
do
    for data in ${datasets[@]}
    do
        for model in ${models[@]}
        do
            val="configuration=${i}, model=${model}, dataset=${data}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config${i}_${model}_${data}" -w true python ../main.py --dataset ${data} --model ${model} --hidden_dims ${hidden_dims[i]} --heads ${heads[i]} --head_dims ${head_dims[i]} --heads ${heads[i]} --d_a ${d_a[i]} --d_v ${d_v[i]}
        done
    done
done
