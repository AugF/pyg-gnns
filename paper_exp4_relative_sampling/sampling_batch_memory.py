import os
import time

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

algs = ['gcn', 'ggnn', 'gat']
cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
}

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "batch_memory")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for mode in ['cluster', 'graphsage']:
    for data in ['amazon-computers', 'flickr']:
        for alg in algs:
            if mode == 'cluster':
                cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_memory.py --mode {} --model {} --data {} --epochs 50 --batch_partitions {} --json_path {}"
                for cs in cluster_batchs:
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(cs)]) + '.json')
                    if os.path.exists(file_path):
                        continue
                    print(file_path)
                    os.system(cmd.format(mode, alg, data, str(cs), file_path))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            else:
                cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_memory.py --mode {} --model {} --data {} --epochs 50 --batch_size {} --json_path {}"
                for gs in graphsage_batchs[data]:
                    file_path = os.path.join(dir_path, '_'.join([mode, alg, data, str(gs)]) + '.json')
                    if os.path.exists(file_path):
                        continue
                    print(file_path)
                    os.system(cmd.format(mode, alg, data, str(gs), file_path))
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        