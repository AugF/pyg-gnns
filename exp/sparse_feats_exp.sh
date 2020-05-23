#!/usr/bin/env bash
# for features and graph experiment
dir_feats="sparse_feats_exp" # datasets: {dataname}_{feat_dims}_{ratio}

if [ ! -d $dir_feats ]
then
    mkdir $dir_feats
fi


echo "begin input features sparse experiment..."
for name in com-amazon
do
    for model in gcn ggnn gat gaan
    do
        for ratio in 5 10 20 50 # features dims fix at 500, ratio changes
        do
            val="model=${model}, dataset=${name}_500_${ratio}, feats_dims=500, ratio=${ratio}%"
            echo $val
            nsys profile -t cuda,osrt,nvtx -o "${dir_feats}/${model}_${name}_500_${ratio}" -w true python ../main.py --dataset "${name}_500_${ratio}" --model $model
        done
        for dims in 250 750 1000 1250 # ratio fix at 0.2, features dims changes
        do
            val="model=${model}, dataset=${name}_${dims}_20, feats_dims=${dims}, ratio=20%"
            echo $val
            nsys profile -t cuda,osrt,nvtx -o "${dir_feats}/${model}_${name}_${dims}_20" -w true python ../main.py --dataset "${name}_${dims}_20" --model $model
        done
    done
done


