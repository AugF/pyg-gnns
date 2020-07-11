date
cd /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp 
echo "begin sampling"
bash sampling_batch_time.sh >>batch_time.log 2>&1
date
bash sampling_batch_time_full.sh >>batch_time.log 2>&1
date

echo "begin exp degrees"
cd /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/graph_degrees_exp
bash exp_avg_degrees.sh >>exp_avg_degrees_7_11.log 2>&1
date
echo "begin memory degrees"
bash memory_avg_degrees.sh >>memory_avg_degrees_7_11.log 2>&1
date

echo "begin memory edges"
cd /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/graph_nodes_exp
bash memory_edges.sh >>memory_edges_7_11.log 2>&1
echo "end..."
date
