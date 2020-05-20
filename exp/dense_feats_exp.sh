
dir_feats="dense_feats_exp"

if [ ! -d $dir_feats ]
then
    mkdir $dir_feats
fi

echo "begin dense feats experiment..."

for name in flickr com-amazon
do
    for model in gcn ggnn gat gaan
    do
        for d in 16 32 64 128 256 512
        do
            nsys profile -t cuda,nvtx,osrt -o "${dir_feats}/${model}_${name}_${d}" -w true python ../main.py --model $model --dataset "${name}_${d}"
        done
    done
done
