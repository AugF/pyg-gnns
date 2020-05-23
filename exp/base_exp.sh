#!/usr/bin/env bash
# experiment for setting configuration
dir_base="base_exp"
datasets=(flickr com-amazon reddit com-lj)
models=(gcn ggnn gat gaan)
hds=(64 32 16 8)
hs=(8 4 2 1)

echo "begin base experiment..."
for hd in ${hds[@]}
do
    for h in ${hs[@]}
    do
        for data in ${datasets[@]}
        do
            for model in ${models[@]}
            do
                val="model=${model}, data=${data}, hidden_dims=${hd}, heads=${h}"
                echo ${val}
                t=`expr $hd / $h`
                python ../main.py --dataset $data --model $model --hidden_dims $hd --heads $h --head_dims $t --d_a $t --d_v $t
            done
        done
    done
done