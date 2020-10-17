import os
import time

nodes = ['05k', '1k', '2k', '4k', '6k', '8k', '10k', '15k']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
degrees = 4

dir_path = 'small_graph_train_time'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for alg in algs:
    for i in range(50):
        for ns in nodes:
            data = 'graph_' + str(ns) + '_4_' + str(i) 
            print(data)
            cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_full.py --model {} --data {} --epochs 50 >>{} 2>&1"
            file_path = os.path.join(dir_path, '_'.join([alg, data]) + '.log')
            if os.path.exists(file_path):
                continue
            print(file_path)
            os.system(cmd.format(alg, data, file_path))
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        