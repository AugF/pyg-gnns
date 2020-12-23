#!/usr/bin/env bash
dir_config="dir_gaan_inference_qdrep"
dir_sqlite="dir_gaan_inference_sqlite"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
fi

datasets=(amazon-photo pubmed amazon-computers flickr com-amazon)
models=(gaan)
ds=(8 16 32 64 128 256)
heads=(1 2 4 8 16)
hds=(16 32 64 128 256 512 1024 2048)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        # d_a, d_v, d_m,  heads=4, hidden_dims=64
        for hd in ${ds[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, heads=4, d_a, d_v, d_m=${hd}, hidden_dims=64"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_4_${hd}_64.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_4_${hd}_64" -w true python ../main_inference.py --dataset ${data} --model ${model} --d_v ${hd} --d_a ${hd} --d_m ${hd} --heads 4 --hidden_dims 64
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_4_${hd}_64.qdrep" "${dir_sqlite}/config0_${model}_${data}_4_${hd}_64.sqlite"
        done

        # hidden_dims
        for hd in ${hds[@]}
        do
            val="configuration=${i}, model=${model}, dataset=${data}, heads=4, d_a, d_v, d_m=32, hidden_dims=${hd}"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_4_32_${hd}.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_4_32_${hd}" -w true python ../main_inference.py --dataset ${data} --model ${model} --hidden_dims ${hd} --heads 4 --d_v 32 --d_a 32 --d_m 32
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_4_32_${hd}.qdrep" "${dir_sqlite}/config0_${model}_${data}_4_32_${hd}.sqlite"
        done
        
        # heads
        for h in ${heads[@]}
        do
            val="configuration=0, model=${model}, dataset=${data}, heads=${h}, d_a, d_v, d_m=32, hidden_dims=64"
            echo ${val}
            if [ -f "${dir_config}/config0_${model}_${data}_${h}_32_64.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_${model}_${data}_${h}_32_64" -w true python ../main_inference.py --dataset ${data} --model ${model} --d_m 32 --d_v 32 --d_a 32 --heads ${h} --hidden_dims 64
            nsys-exporter -s "${dir_config}/config0_${model}_${data}_${h}_32_64.qdrep" "${dir_sqlite}/config0_${model}_${data}_${h}_32_64.sqlite"
        done
    done
done
