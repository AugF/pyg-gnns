date
cd "${dir_path}/paper_exp4_relative_sampling"
echo "exp4 "
bash sampling_batch_full_memory.sh >sampling_batch_full_memory.log 2>&1
date
bash sampling_batch_full_time.sh >sampling_batch_full_time.log 2>&1
date
python sampling_batch_memory.py
date
python sampling_batch_train_time_stack.py
date