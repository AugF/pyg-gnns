export LD_LIBRARY_PATH=/home/wangzhaokang/anaconda3/envs/pyg1.5/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
echo "run_gcn_ggnn"
bash gcn_ggnn_memory.sh >gcn_ggnn_memory.log 2>&1
echo "run exp: gaan"
bash gaan_exp.sh >gaan_exp.log 2>&1
echo "run exp: gat"
bash gat_exp.sh >gat_exp.log 2>&1
echo "run exp: gcn_ggnn"
bash gcn_ggnn_exp.sh >gcn_ggnn_exp.log 2>&1
