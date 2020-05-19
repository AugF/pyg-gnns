#!/usr/bin/env bash
# for features and graph experiment
dir_feats="feats_exp" # datasets: {dataname}_{feat_dims}_{ratio}
dir_graph="graph_exp" # datasets: graph_{nodes}_{degree}

echo "begin input features experiment..."
for name in flickr com-amazon
do
    for model in gcn ggnn gat gaan
    do
        for ratio in 5 10 20 50 # features dims fix at 500, ratio changes
        do
            val="model=${model}, dataset=${name}_500_${ratio}, feats_dims=500, ratio=${ratio}%"
            echo $val
            echo $val >> "${dir_feats}.log"
            nsys profile -t cuda, osrt, nvtx -o "${dir_feats}/${model}_${name}_500_${ratio}" -w true python ../main.py --dataset "${name}_500_${ratio}" --model $model >> "${dir_feats}.log"
        done
        for dims in 250 750 1000 1250 # ratio fix at 0.2, features dims changes
        do
            val="model=${model}, dataset=${name}_${dims}_20, feats_dims=${dims}, ratio=20%"
            echo $val
            echo $val >> "${dir_feats}.log"
            nsys profile -t cuda, osrt, nvtx -o "${dir_feats}/${model}_${name}_${dims}_20" -w true python ../main.py --dataset "${name}_${dims}_20" --model $model >> "${dir_feats}.log"
        done
    done
done


echo "begin graph scalablity experiment..."
for model in gcn ggnn gat gaan
    do
        for nodes in 100k 5k 1m 5m # degree fix at 25, nodes changes
        do
            val="model=${model}, dataset=graph_${nodes}_25, nodes=${dims}, degree=25"
            echo ${val}
            echo ${val} >> "${dir_graph}.log"
            nsys profile -t cuda, osrt, nvtx -o "${dir_graph}/${model}_graph_${nodes}_25" -w true python ../main.py --dataset "graph_${nodes}_25" --model ${model} >> "${dir_graph}.log"
        done
        for degree in 10 50 75 100 # features dims fix at 500, ratio changes
        do
            val="model=${model}, dataset=graph_500k_${degree}, nodes=500k, degree=${degree}"
            echo ${val}
            echo ${val} >> "${dir_graph}.log"
            nsys profile -t cuda, osrt, nvtx -o "${dir_feats}/${model}_graph_500k_${degree}" -w true python ../main.py --dataset "graph_500k_${degree}" --model ${model} >> "${dir_graph}.log"
        done
    done
done

