#!/usr/bin/env bash
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

datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)
models=(gcn ggnn gat gaan)
vars=(5 10 20 50)

for model in ${models[@]}
do
    for data in ${datasets[@]}
    do
        for var in ${vars[@]}
        do
             qdrep_path="${dir_config}/config0_${model}_${data}_500_${var}"
            if [ -f "${qdrep_file}.qdrep" ]; then # 断点续传
                continue
            fi
            val="configuration=0, model=${model}, dataset=${data}, var=${var}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_500_${var}" -w true python ../main.py --dataset "${data}_500_${var}" --model ${model}
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_500_${var}.qdrep" "${dir_sqlite}/config0_${model}_${data}_500_${var}.sqlite"
        done
    done
done
