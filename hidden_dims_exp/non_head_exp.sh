#!/usr/bin/env bash
dir_config="dir_qdrep"
dir_sqlite="dir_sqlite"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
fi

datasets=(amazon-photo amazon-computers)
models=(gcn)
hds=(1024)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for hd in ${hds[@]}
        do
            val="configuration=${i}, model=${model}, dataset=${data}, hidden_dims=${hd}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${hd}.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_${hd}" -w true python ../main.py --dataset ${data} --model ${model} --hidden_dims ${hd}
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_${hd}.qdrep" "${dir_sqlite}/config0_${model}_${data}_${hd}.sqlite"
        done
    done
done
