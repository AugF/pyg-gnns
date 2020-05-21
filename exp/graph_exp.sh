#!/usr/bin/env bash
# for features and graph experiment
dir_graph="graph_exp" # datasets: graph_{nodes}_{degree}


if [ ! -d $dir_graph ]
then
    mkdir $dir_graph
fi

echo "begin graph scalablity experiment..."
for model in gaan
    do
        for nodes in 1k 25k 50k 75k 100k  # degree fix at 25, nodes changes
        do
            val="model=${model}, dataset=graph_${nodes}_25, nodes=${nodes}, degree=25"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_graph}/${model}_graph_${nodes}_25" -w true python ../main.py --dataset "graph_${nodes}_25" --model ${model}
        done
        for degree in 10 50 75 100 # nodes fix at 50k, degree changes
        do
            val="model=${model}, dataset=graph_50k_${degree}, nodes=50k, degree=${degree}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_graph}/${model}_graph_50k_${degree}" -w true python ../main.py --dataset "graph_50k_${degree}" --model ${model}
        done
    done
done

