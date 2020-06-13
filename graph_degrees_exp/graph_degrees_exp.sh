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
degrees=(10 25 50 75 100)

for model in ${models[@]}
do
    for ds in ${degrees[@]}
    do
        qdrep_path="${dir_config}/config0_${model}_graph_50k_${ds}"
        if [ -f "${qdrep_file}.qdrep" ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_${ns}k_25"
        echo ${val}
        nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_graph_50k_${ds}" -w true python "${base_path}/../main.py" --dataset "graph_50k_${ds}" --model ${model}
        nsys-exporter -s "${dir_config}/config0_${model}_graph_50k_${ds}.qdrep" "${dir_sqlite}/config0_${model}_graph_50k_${ds}.sqlite"
    done
done
