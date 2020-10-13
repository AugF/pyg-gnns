base_path="/home/wangzhaokang/wangyunpan/pyg-gnns/Technical-report/sample-technique-motivation/code"

cd "${base_path}/ogbn-products"
echo "ogbn-products test..."
echo "cluster_gcn test..."
date
python cluster_gcn.py 1>>run_base.log 2>&1
date
echo "graph_saint test..."
date
python graph_saint.py 1>>run_base.log 2>&1
date
echo "neighbor_sampling test..."
date
python neighbor_sampling.py 1>>run_base.log 2>&1
date

cd "${base_path}/ogbn-mag"
echo "ogbn-mag test..."
echo "cluster_gcn test..."
date
python cluster_gcn.py 1>>run_base.log 2>&1
date
echo "graph_saint test..."
date
python graph_saint.py 1>>run_base.log 2>&1
date
echo "neighbor_sampling test..."
date
python neighbor_sampling.py 1>>run_base.log 2>&1
date
echo "rgcn test..."
date
python neighbor_sampling.py 1>>run_base.log 2>&1
date

cd "${base_path}/ogbl-citation"
echo "ogbn-citation test..."
echo "cluster_gcn test..."
date
python cluster_gcn.py 1>>run_base.log 2>&1
date
echo "graph_saint test..."
date
python graph_saint.py 1>>run_base.log 2>&1
date