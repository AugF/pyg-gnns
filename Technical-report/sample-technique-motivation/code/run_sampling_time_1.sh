base_path="/home/wangzhaokang/wangyunpan/pyg-gnns/Technical-report/sample-technique-motivation/code"

# ogbn-mag
cd "${base_path}/ogbn-mag"
echo "ogbn-mag test..."

cluster_sizes=(50 150 300 500 1250 2500)
batch_sizes=(19397 58192 116384 193974 484935 969871)

echo "cluster_gcn test..."
for batch_size in ${cluster_sizes[@]}
do
    echo "batch_size=${batch_size}"
    if [ -f "cluster_gcn_${batch_size}.npy" ]; then 
        continue
    fi
    date
    python -u cluster_gcn.py --batch_size ${batch_size} --device 1 1>>"cluster_gcn_${batch_size}".log 2>&1
    date
done

echo "graph_saint test..."
for batch_size in ${batch_sizes[@]}
do
    echo "batch_size=${batch_size}"
    if [ -f "graph_saint_${batch_size}.npy" ]; then 
        continue
    fi
    date
    python -u graph_saint.py --batch_size ${batch_size} --device 1 1>>"graph_saint_${batch_size}".log 2>&1
    date
done

echo "neighbor_sampling test..."
for batch_size in ${batch_sizes[@]}
do
    echo "batch_size=${batch_size}"
    if [ -f "neighbor_sampling_${batch_size}.npy" ]; then 
        continue
    fi
    date
    python -u neighbor_sampling.py --batch_size ${batch_size} --device 1 1>>"neighbor_sampling_${batch_size}".log 2>&1
    date
done