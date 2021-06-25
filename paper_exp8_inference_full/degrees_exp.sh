#!/usr/bin/env bash
export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
dir_config="dir_degrees_qdrep"
dir_sqlite="dir_degrees_sqlite"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
fi

models=(gcn gat gaan)
degrees=(3 6 10 15 20 25 30 50)

for model in ${models[@]}
do
    for ds in ${degrees[@]}
    do
        qdrep_path="${dir_config}/config0_${model}_graph_50k_${ds}"
        if [ -f "${qdrep_file}.sqlite" ]; then # 断点续传
            continue
        fi
        val="configuration=0, model=${model}, dataset=graph_50k_${ds}"
        echo ${val}
        # nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_graph_50k_${ds}" -w true python ../main_inference.py --dataset "graph_50k_${ds}" --model ${model}
        nsys-exporter -s "${dir_config}/config0_${model}_graph_50k_${ds}.qdrep" "${dir_sqlite}/config0_${model}_graph_50k_${ds}.sqlite"
    done
done
