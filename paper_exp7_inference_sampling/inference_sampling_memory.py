import os
import time

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
# small_datasets =  ['amazon-computers', 'flickr']

algs = ['gcn', 'ggnn', 'gat', 'gaan']

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "inference_sampling_memory_2048")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

sh_commands = []
for alg in algs:
    for data in small_datasets:
        file_path = os.path.join(dir_path, '_'.join([alg, data]) + '.json')
        if os.path.exists(file_path):
            continue
        print(file_path)
        cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_inference_sampling.py --batch_size 2048 --device cuda:0 --model {} --data {} --epochs 10 --infer_json_path {}"
        sh_commands.append(cmd.format(alg, data, file_path))

with open("inference_sampling_memory_2048.sh", "w") as f:
    for sh in sh_commands:
        f.write(sh + '\n')