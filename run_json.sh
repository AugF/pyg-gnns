modes=(cluster graphsage)
models=(gcn ggnn gat gaan)
datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for mode in ${modes[@]}
        do
            nsys profile -t cuda,osrt,nvtx -o "/data/wangzhaokang/wangyunpan/sampling_exp/${model}_${data}_${mode}.json" python main_sampling.py --mode $mode --model $model --data $data --epochs 2
        done
    done
done
