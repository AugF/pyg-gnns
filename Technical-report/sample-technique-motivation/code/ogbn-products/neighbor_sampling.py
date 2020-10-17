# Reaches around 0.7870 ± 0.0036 test accuracy.

import os.path as osp
import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-MAG (SAGE)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-products')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
logger = Logger(args.runs, args)

data = dataset[0]

train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=args.batch_size,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, num_layers=args.num_layers)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    # pbar = tqdm(total=train_idx.size(0))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    
    sampling_time, to_time, train_time = 0.0, 0.0, 0.0
    loader_iter = iter(train_loader)
    
    while True:
        try:
            t0 = time.time()
            batch_size, n_id, adjs = next(loader_iter)
            t1 = time.time()
            adjs = [adj.to(device) for adj in adjs]
            t2 = time.time()

            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            # pbar.update(batch_size)
            train_time += time.time() - t2
            to_time += t2 - t1
            sampling_time += t1 - t0   
        except StopIteration:
            break
    # pbar.close()

    loss = total_loss / len(train_loader)

    return loss, sampling_time, to_time, train_time


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

avg_sampling_time, avg_to_time, avg_train_time = 0.0, 0.0, 0.0

test() # Test if inference on GPU succeeds.
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        t0 = time.time()
        loss, sampling_time, to_time, train_time = train(epoch)

        torch.cuda.empty_cache()
        avg_sampling_time += sampling_time
        avg_to_time += to_time
        avg_train_time += train_time
        
        if epoch % 2 == 0:
            result = test()
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}%, '
                f'Test: {100 * test_acc:.2f}%, '
                f'Time: {time.time() - t0}s')
    logger.print_statistics(run)

avg_sampling_time /= args.runs * args.epochs
avg_to_time /= args.runs * args.epochs
avg_train_time /= args.runs * args.epochs
print(f'Avg_sampling_time: {avg_sampling_time}s, '
        f'Avg_to_time: {avg_to_time}s, ',
        f'Avg_train_time: {avg_train_time}s')

logger.print_statistics()
logger.save("neigbor_sampling_" + str(args.batch_size))