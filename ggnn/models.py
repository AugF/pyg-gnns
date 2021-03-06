import torch

import torch.nn.functional as F
from torch.nn import Parameter, Module
from ggnn.layers import GatedGraphConv
from tqdm import tqdm

from inits import glorot
from utils import nvtx_push, nvtx_pop, log_memory


class GGNN(Module):
    """
    GGNN layer
    """
    def __init__(self, layers, n_features, n_classes, hidden_dims, gpu=False, device=None, flag=False):
        super(GGNN, self).__init__()
        self.n_features, self.n_classes = n_features, n_classes
        self.layers, self.hidden_dims = layers, hidden_dims
        self.gpu = gpu
        self.device = device
        self.flag = flag

        self.weight_in = Parameter(torch.Tensor(n_features, hidden_dims))
        self.weight_out = Parameter(torch.Tensor(hidden_dims, n_classes))
        self.convs = torch.nn.ModuleList([GatedGraphConv(out_channels=hidden_dims, num_layers=layers, gpu=gpu, flag=flag, device=device)])
        glorot(self.weight_in)
        glorot(self.weight_out)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adjs):
        device = torch.device(self.device)
        nvtx_push(self.gpu, "input-transform")
        x = torch.matmul(x, self.weight_in)
        
        nvtx_pop(self.gpu)
        log_memory(self.flag, device, "input-transform")

        x = self.convs[0](x, adjs)
        nvtx_push(self.gpu, "output-transform")
        x = torch.matmul(x, self.weight_out)
        nvtx_pop(self.gpu)
        log_memory(self.flag, device, "output-transform")
        return x

    def inference(self, x_all, subgraph_loader):
        device = torch.device(self.device)
        # pbar = tqdm(total=x_all.size(0) * self.layers)
        # pbar.set_description('Evaluating')

        x_all = torch.matmul(x_all.to(device), self.weight_in) # 尽最大可能第键槽内存
        
        x_all = self.convs[0].inference(x_all.cpu(), subgraph_loader)
        x_all = torch.matmul(x_all.to(device), self.weight_out)
        # pbar.close()

        return x_all.cpu()
    
    def __repr__(self):
        return '{}(layers={}, n_features={}, n_classes={}, hidden_dims={}, gpu={})'.format(
            self.__class__.__name__, self.layers, self.n_features, self.n_classes,
            self.hidden_dims, self.gpu) + '\n' + str(self.convs)

