import os
import time

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

re_percents = [0.01, 0.03, 0.06, 0.10, 0.25, 0.50]


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "batch_more_memory")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


for mode in ['cluster', 'graphsage']:
    for data in ['amazon-computers', 'flickr', 'yelp']:
        for alg in ['gcn', 'gat']:
            if mode == 'cluster':
                cmd = "python /mnt/data/wangzhaokang/wangyunpan/pyg-gnns/main_sampling_more_memory.py --gpu 1 --mode {} --model {} --data {} --epochs 50 --batch_partitions {} --real_path {}"
                for rs in [40, 60, 80]:
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(rs)]) + '.csv')
                    if os.path.exists(file_path):
                        continue
                    print(file_path)
                    os.system(cmd.format(mode, alg, data, str(rs), file_path))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            else:
                cmd = "python /mnt/data/wangzhaokang/wangyunpan/pyg-gnns/main_sampling_more_memory.py --gpu 1 --mode {} --model {} --data {} --epochs 50 --batch_partitions {} --real_path {}"
                for rs in [1024, 2048, 4096]:
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(rs)]) + '.csv')
                    if os.path.exists(file_path):
                        continue
                    print(file_path)
                    os.system(cmd.format(mode, alg, data, str(rs), file_path))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))