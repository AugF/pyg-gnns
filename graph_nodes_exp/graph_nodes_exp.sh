#!/usr/bin/env bash
base_path=$(cd `dirname $0`; pwd)
dir_config="${base_path}/dir_qdrep"
dir_sqlite="${base_path}/dir_sqlite"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
fi

models=(gcn ggnn gat gaan)
nodes=(1 25 50 75 100)

for model in ${models[@]}
do
    for ns in ${nodes[@]}
    do
        qdrep_path="${dir_config}/config0_${model}_graph_${ns}k_25"
        if [ -f "${qdrep_file}.qdrep" ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_${ns}k_25"
        echo ${val}
        nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_graph_${ns}k_25" -w true python "${base_path}/../main.py" --dataset "graph_${ns}k_25" --model ${model}
        nsys-exporter -s "${dir_config}/config0_${model}_graph_${ns}k_25.qdrep" "${dir_sqlite}/config0_${model}_graph_${ns}k_25.sqlite"
    done
done
