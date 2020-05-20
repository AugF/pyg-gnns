#!/usr/bin/env bash
# for features and graph experiment
dir_graph="graph_exp" # datasets: graph_{nodes}_{degree}


if [ ! -d $dir_graph ]
then
    mkdir $dir_graph
fi

echo "begin graph scalablity experiment..."
for model in gcn ggnn gat gaan
    do
        for nodes in 100k 500k 1m # degree fix at 25, nodes changes
        do
            val="model=${model}, dataset=graph_${nodes}_25, nodes=${nodes}, degree=25"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_graph}/${model}_graph_${nodes}_25" -w true python ../main.py --dataset "graph_${nodes}_25" --model ${model}
        done
        for degree in 10 50 75 100 # features dims fix at 500, ratio changes
        do
            val="model=${model}, dataset=graph_500k_${degree}, nodes=500k, degree=${degree}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_graph}/${model}_graph_500k_${degree}" -w true python ../main.py --dataset "graph_500k_${degree}" --model ${model}
        done
    done
done

