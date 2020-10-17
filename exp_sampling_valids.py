import os
import time

algs = ['gaan']
datasets = ['amazon-computers', 'flickr']
modes = ['graphsage', 'cluster']

dir_path = "sampling_valids_log"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# algorithms experiment
# for alg in algs:
#     cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_inductive.py --model {} --data pubmed --epochs 500 >>{} 2>&1"
#     log_file = dir_path + "/" + alg + "_pubmed_full.log"
#     if not os.path.exists(log_file):
#         os.system(cmd.format(alg, log_file))
#     print(alg, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
#     for mode in modes:
#         batch_cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_batch.py --model {} --data pubmed --epochs 50 --mode {} >>{} 2>&1"
#         batch_log_file = dir_path + "/" + alg + "_pubmed_" + mode + "_batch.log"
#         if not os.path.exists(batch_log_file):
#             os.system(batch_cmd.format(alg, mode, batch_log_file))
#         epoch_cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_epoch.py --model {} --data pubmed --epochs 50 --mode {} >>{} 2>&1"
#         epoch_log_file = dir_path + "/" + alg + "_pubmed_" + mode + "_epoch.log"
#         if not os.path.exists(epoch_log_file):
#             os.system(epoch_cmd.format(alg, mode, epoch_log_file))
#         print(alg, mode, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

for data in datasets:
    cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_inductive.py --model gcn --data {} --epochs 500 >>{} 2>&1"
    log_file = dir_path + "/gcn_" + data + "_full.log"
    if not os.path.exists(log_file):
        os.system(cmd.format(data, log_file))
    print(data, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for mode in modes:
        batch_cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_batch.py --model gcn --data {} --epochs 50 --mode {} >>{} 2>&1"
        batch_log_file = dir_path + "/gcn_" + data + "_" + mode + "_batch.log"
        if not os.path.exists(batch_log_file):
            os.system(batch_cmd.format(data, mode, batch_log_file))
        epoch_cmd = "python /home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/main_sampling_epoch.py --model gcn --data {} --epochs 50 --mode {} >>{} 2>&1"
        epoch_log_file = dir_path + "/gcn_" + data + "_" + mode + "_epoch.log"
        if not os.path.exists(epoch_log_file):
            os.system(epoch_cmd.format(data, mode, epoch_log_file))
        print(data, mode, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  
            
        