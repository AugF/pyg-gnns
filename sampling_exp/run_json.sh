modes=(cluster graphsage)
models=(gcn ggnn gat gaan)
datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for mode in ${modes[@]}
        do
            file_path="/data/wangzhaokang/wangyunpan/sampling_exp/${model}_${data}_${mode}.json"
            if [ -f ${file_path} ]; then # 断点续传
                continue
            fi
            python ../main_sampling.py --mode $mode --model $model --data $data --json_path $file_path --epochs 2 
        done
    done
done
