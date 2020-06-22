dir_path="/data/wangzhaokang/wangyunpan/sampling_exp1"
modes=(graphsage)
models=(gcn ggnn gat gaan)
datasets=(amazon-photo pubmed amazon-computers coauthor-physics flickr com-amazon)

for data in ${datasets[@]}
do
    for model in ${models[@]}
    do
        for mode in ${modes[@]}
        do
            file_path="${dir_path}/${model}_${data}_${mode}"
            if [ -f "${qdrep_file}.qdrep" ]; then # 断点续传
                continue
            fi
            nsys profile -t cuda,osrt,nvtx -o "${dir_path}/${model}_${data}_${mode}" -w true python main_sampling.py --mode $mode --model $model --data $data --epochs 2
            nsys-exporter -s "${dir_path}/${model}_${data}_${mode}.qdrep" "${dir_path}/${model}_${data}_${mode}.sqlite"
        done
    done
done
