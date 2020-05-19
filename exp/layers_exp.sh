#!/usr/bin/env bash
# for layers experiment
dir_layer="layers_exp"

if [ ! -d $dir_layer ]
then
    mkdir $dir_layer
fi

echo "begin layers experiment..."
for data in flickr com-amazon
do
    for model in gcn ggnn
    do
        for l in 16 32 64 128 256 512 1024 # hidden_dims changes
        do
            val="model=${model}, dataset=${data}, layers=${l}"
            echo ${val}
            echo ${val} >> "${dir_layer}.log"
            nsys profile -t cuda,osrt,nvtx -o "${dir_layer}/${model}_${data}_${l}" -w true python ../main.py --dataset $data --model $model --layers $l >> "${dir_layer}.log"
        done
    done
done
