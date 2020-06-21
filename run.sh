date
bash run_json.sh >>run_json.log 2>&1
date
echo "run_json end"
bash run_nsys.sh >>run_nsys.log 2>&1
date
echo "run_nsys end"
bash run_time.sh >>run_time.log 2>&1
echo "run_time end"
date