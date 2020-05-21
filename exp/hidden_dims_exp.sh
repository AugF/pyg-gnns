#!/usr/bin/env bash
# for hidden feature dimension experiment
dir_head="hidden_dims_exp/multi_head"
dir_non_head="hidden_dims_exp/non_multi_head"

if [ ! -d $dir_head ]
then
    mkdir -p $dir_head
fi

if [ ! -d $dir_non_head ]
then
    mkdir -p $dir_non_head
fi

# multi-head
echo "begin multi-head experiment..."
for data in flickr com-amazon
do
    for model in gat gaan
    do
        for hds in 8 16 32 64 128 256 # heads fix at 4, head_dims changes
        do
            val="model=${model}, dataset=${data}, heads=4, head_dims=${hds}"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_head}/${model}_${data}_4_${hds}" -w true python ../main.py --dataset $data --model $model --heads 4 --head_dims $hds --d_a $hds --d_v $hds
        done
        for head in 1 2 8 16 # head_dims fix at 64, heads changes
        do
            val="model=${model}, dataset=${data}, heads=${head}, head_dims=64"
            echo ${val}
            nsys profile -t cuda,osrt,nvtx -o "${dir_head}/${model}_${data}_${head}_64" -w true python ../main.py --dataset $data --model $model --heads $head --head_dims 64 --d_a 64 --d_v 64
        done
    done
done

echo "begin non-multi-head experiment..."
# non-multi-head
for data in flickr com-amazon
do
    for model in gcn ggnn
    do
        for hds in 16 32 64 128 256 512 1024 # hidden_dims changes
        do
            val="model=${model}, dataset=${data}, hidden_dims=${hds}"
            echo $val
            nsys profile -t cuda,osrt,nvtx -o "${dir_non_head}/${model}_${data}_${hds}" -w true python ../main.py --dataset $data --model $model --hidden_dims $hds
        done
    done
done
