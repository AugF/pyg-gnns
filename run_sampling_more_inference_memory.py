import os
import time

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

re_percents = [1024, 2048, 4096, 8192]


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "batch_more_inference_memory")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


for data in ['amazon-computers', 'amazon-photo']:
    for alg in ['gcn', 'gat', 'ggnn', 'gaan']:
        cmd = "python /mnt/data/wangzhaokang/wangyunpan/pyg-gnns/main_sampling_more_inference_memory.py --gpu 1 --mode {} --data {} --epochs 50 --infer_batch_size {} --real_path {}"
        # for rs in re_percents:
        for rs in [16384]:
            file_path = os.path.join(dir_path, '_'.join([alg, data, str(rs)]) + '.csv')
            if os.path.exists(file_path):
                continue
            print(file_path)
            real_cmd = cmd.format(alg, data, str(rs), file_path)
            print(real_cmd)
            os.system(real_cmd)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))