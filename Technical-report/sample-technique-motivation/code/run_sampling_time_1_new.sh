base_path="/home/wangzhaokang/wangyunpan/pyg-gnns/Technical-report/sample-technique-motivation/code"

# ogbn-products
cd "${base_path}/ogbn-products"
echo "ogbn-products test..."

batch_sizes=(24490 73470 146941 244902 612257 1224514)

echo "neighbor_sampling test..."
for batch_size in ${batch_sizes[@]}
do
    echo "batch_size=${batch_size}"
    date
    python -u neighbor_sampling.py --batch_size ${batch_size} --device 1 1>>"neighbor_sampling_${batch_size}".log 2>&1
    date
done
