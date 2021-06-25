import torch
import time

import torch.nn.functional as F
from torch.nn import Parameter, Module
from gat.layers import GATConv
from tqdm import tqdm

from inits import glorot
from utils import nvtx_push, nvtx_pop, log_memory


class GAT(Module):
    """
    GAT model
    dropout, negative_slop set: https://github.com/Diego999/pyGAT/blob/master/train.py
    """
    def __init__(self, layers, n_features, n_classes, head_dims,
                 heads, dropout=0.6, attention_dropout=0.6, negative_slop=0.2,
                 gpu=False, device="cpu", flag=False, infer_flag=False, sparse_flag=False):
        super(GAT, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.head_dims, self.heads = layers, head_dims, heads
        self.dropout, self.negative_slop = dropout, negative_slop
        self.gpu = gpu
        self.device = device
        self.flag, self.infer_flag = flag, infer_flag
        self.sparse_flag = sparse_flag        

        in_shapes = [n_features] + [head_dims * heads] * (layers - 1)
        out_shapes = [head_dims] * (layers - 1) + [n_classes]
        head_shape = [heads] * (layers - 1) + [1]
        self.convs = torch.nn.ModuleList(
            [
                GATConv(in_channels=in_shapes[layer], out_channels=out_shapes[layer],
                        heads=head_shape[layer], dropout=attention_dropout, negative_slope=negative_slop,
                        gpu=gpu, concat=layer != layers - 1)
                for layer in range(layers)
             ]
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, x, adjs):
        device = torch.device(self.device)
        
        if isinstance(adjs, list):
            for i, (edge_index, _, size) in enumerate(adjs):
                nvtx_push(self.gpu, "layer" + str(i))
                if not self.sparse_flag:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i]((x, x[:size[1]]), edge_index)
                if i != self.layers - 1:
                    x = F.elu(x)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
        else:
            for i in range(self.layers):
                nvtx_push(self.gpu, "layer" + str(i))
                if not self.sparse_flag:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i](x, adjs)
                if i != self.layers - 1:
                    x = F.elu(x)
                nvtx_pop(self.gpu)
                log_memory(self.flag, device, 'layer' + str(i))
                
        return x
    
    def inference(self, x_all, subgraph_loader, df=None):
        device = torch.device(self.device)
        flag = self.infer_flag
        
        sampling_time, to_time, train_time = 0.0, 0.0, 0.0
        total_batches = len(subgraph_loader)

        log_memory(flag, device, 'inference start')     
        for i in range(self.layers):
            log_memory(flag, device, f'layer{i} start')

            xs = []
            loader_iter = iter(subgraph_loader)
            while True:
                try:
                    torch.cuda.reset_max_memory_allocated(device)
                    torch.cuda.empty_cache()
                    current_memory = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]
                
                    et0 = time.time()
                    batch_size, n_id, adj = next(loader_iter)
                    log_memory(flag, device, 'batch start')  
                                   
                    et1 = time.time()
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    log_memory(flag, device, 'to end')
                                       
                    et2 = time.time()
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    # x_target = x[:size[1]]
                    x = self.convs[i]((x, x[:size[1]]), edge_index)
                    if i != self.layers - 1:
                        x = F.elu(x)
                    xs.append(x.cpu())
                    log_memory(flag, device, 'batch end')                    
                    
                    sampling_time += et1 - et0
                    to_time += et2 - et1
                    train_time += time.time() - et2
                    if df is not None:
                        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
                        df['memory'].append(memory)
                        df['diff_memory'].append(memory - current_memory)
                except StopIteration:
                    break
            
            x_all = torch.cat(xs, dim=0)
            
        sampling_time /= total_batches
        to_time /= total_batches
        train_time /= total_batches
        
        log_memory(flag, device, 'inference end') 
        print(f"avg_batch_train_time: {train_time}, avg_batch_sampling_time:{sampling_time}, avg_batch_to_time: {to_time}")
        return x_all
    
    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, head_dims={}, heads={}' \
               ', dropout={}, negative_slop={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes, self.head_dims,
            self.heads, self.dropout, self.negative_slop, self.gpu) + '\nLayer(dropout->conv->elu)\n' + str(self.convs)



