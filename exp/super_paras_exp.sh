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
    for model in gcn ggnn gat gaan
    do
        for l in 1 2 3 4 5 #layers changes
        do
            val="model=${model}, dataset=${data}, layers=${l}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_layer}/${model}_${data}_${l}" -w true python ../main.py --dataset $data --model $model --layers $l
        done
    done
done
        exit
