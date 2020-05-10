"""
todo: fix
"""
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score

import os.path as osp

import torch
import torch.nn.functional as F
from gaan.layers import GaANConv

# PPI数据集测试
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
# todo 注意到这里下载的文件应该是不能用的
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GaANConv(in_channels=train_dataset.num_features, out_channels=256,
                              heads=8, d_v=32, d_a=24, d_m=64, dropout=0.1,
                              negative_slope=0.1)
        self.conv2 = GaANConv(in_channels=256, out_channels=train_dataset.num_classes,
                              heads=8, d_v=32, d_a=24, d_m=64, dropout=0.1,
                              negative_slope=0.1)
        self.dropout = 0.1

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return x

# 参数
valid_iter = 100
test_iter = 100
max_iter = 100000
lr = 0.01
min_lr = 0.001
decay_patience = 15
early_stopping_patience = 30
lr_decay_factor = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

best_valid_f1 = 0
no_better_valid = 0
is_best = False
cur_lr = 0.01

for epoch in range(max_iter):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))

    if epoch % valid_iter == 0:
        if val_f1 > best_valid_f1:
            is_best = True
            best_valid_f1 = val_f1
            no_better_valid = 0
        else:
            is_best = False
            no_better_valid += 1
            if no_better_valid > early_stopping_patience:
                break
            elif no_better_valid > decay_patience:
                lr -= lr * lr_decay_factor
                if lr >= min_lr:
                    for g in optimizer.param_groups:
                        g['lr'] = lr
