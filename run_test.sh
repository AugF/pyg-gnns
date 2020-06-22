modes=(cluster graphsage)
models=(gcn ggnn gat gaan)
datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for mode in ${modes[@]}
        do
            python main_sampling.py --mode $mode --model $model --data $data --epochs 1
        done
    done
done
