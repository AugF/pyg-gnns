date
python main_inductive.py --epochs 200 --model gat --dataset flickr >out.log 2>&1
date
python main_sampling_epoch.py --mode graphsage --model gat --epochs 50 --data flickr >>out.log 2>&1
date
python main_sampling_batch.py --mode graphsage --model gat --epochs 50 --data flickr >>out.log 2>&1
date
python main_sampling_epoch.py --mode cluster --model gat --epochs 50 --data flickr >>out.log 2>&1
date
python main_sampling_batch.py --mode cluster --model gat --epochs 50 --data flickr >>out.log 2>&1
date