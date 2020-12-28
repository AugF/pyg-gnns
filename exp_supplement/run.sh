export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}

dir_config="dir_qdrep_new"
dir_sqlite="dir_sqlite_new"

if [ ! -d ${dir_config} ]
then
    mkdir -p $dir_config
fi

if [ ! -d $dir_sqlite ]
then
    mkdir -p $dir_sqlite
fi

nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_gat_amazon-computers_4_256" -w true python ../main_paras.py --dataset amazon-computers --model gat --head_dims 256 --heads 4
nsys-exporter -s "${dir_config}/config0_gat_amazon-computers_4_256.qdrep" "${dir_sqlite}/config0_gat_amazon-computers_4_256.sqlite"

# nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_gat_coauthor-physics_4_256" -w true python ../main_paras.py --dataset coauthor-physics --model gat --head_dims 256 --heads 4
# nsys-exporter -s "${dir_config}/config0_gat_coauthor-physics_4_256.qdrep" "${dir_sqlite}/config0_gat_coauthor-physics_4_256.sqlite"

# nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_gat_flickr_4_128" -w true python ../main_paras.py --dataset flickr --model gat --head_dims 128 --heads 4
# nsys-exporter -s "${dir_config}/config0_gat_flickr_4_128.qdrep" "${dir_sqlite}/config0_gat_flickr_4_128.sqlite"

# nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_gat_com-amazon_8_32" -w true python ../main_paras.py --dataset com-amazon --model gat --head_dims 32 --heads 8
# nsys-exporter -s "${dir_config}/config0_gat_com-amazon_8_32.qdrep" "${dir_sqlite}/config0_gat_com-amazon_8_32.sqlite"

# nsys profile -t cuda,osrt,nvtx -o "${dir_config}/config0_gat_flickr_16_32" -w true python ../main_paras.py --dataset flickr --model gat --head_dims 32 --heads 16
# nsys-exporter -s "${dir_config}/config0_gat_flickr_16_32.qdrep" "${dir_sqlite}/config0_gat_flickr_16_32.sqlite"