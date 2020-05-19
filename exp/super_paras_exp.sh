#!/usr/bin/env bash
# lr and weight_decay experiment
dir_sp="sp_exp"
vars=(0.00001 0.0001 0.001 0.01 0.1 0.2 0.4 0.6 0.8)

echo "begin super parameters experiment..."
for name in lr weight_decay
do
    echo "begin ${name} experiment..."
    for var in ${vars[@]}
    do
        val="name=${name}, var=${var}"
        echo $val
        echo $val >> "${dir_sp}.log"
        python ../main.py --model gcn --hidden_dims 16 --dataset com-lj --$name $var >> "${dir_sp}.log"
        python ../main.py --model ggnn --hidden_dims 64 --dataset reddit --$name $var >> "${dir_sp}.log"
        python ../main.py --model gat --hidden_dims 8 --dataset reddit --heads 4 --head_dims 2 --$name $var >> "${dir_sp}.log"
        python ../main.py --model gaan --hidden_dims 64 --dataset com-amazon --$name $var >> "${dir_sp}.log"
    done
done

