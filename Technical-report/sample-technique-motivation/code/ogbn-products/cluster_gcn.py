import argparse
import time

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # pbar = tqdm(total=x_all.size(0) * len(self.convs))
        # pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all


def train(model, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    
    sampling_time, to_time, train_time = 0.0, 0.0, 0.0
    loader_iter = iter(loader)
    while True: 
        try: 
            t0 = time.time()
            data = next(loader_iter)
            t1 = time.time()
            data = data.to(device)
            t2 = time.time()
            if data.train_mask.sum() == 0:
                continue
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
        
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples      

            train_time += time.time() - t2
            to_time += t2 - t1
            sampling_time += t1 - t0      
        except StopIteration:
            break

    return total_loss / total_examples, sampling_time, to_time, train_time


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_partitions', type=int, default=15000)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=args.num_workers)

    model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)

    avg_sampling_time, avg_to_time, avg_train_time = 0.0, 0.0, 0.0
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss, sampling_time, to_time, train_time = train(model, loader, optimizer, device)
            torch.cuda.empty_cache()
            avg_sampling_time += sampling_time
            avg_to_time += to_time
            avg_train_time += train_time
            # if epoch % args.log_steps == 0:
            #     print(f'Run: {run + 1:02d}, '
            #           f'Epoch: {epoch:02d}, '
            #           f'Loss: {loss:.4f}, '
            #           f'Approx Train Acc: {train_acc:.4f}')

            if epoch % 2 == 0:
                result = test(model, data, evaluator, subgraph_loader, device)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%, ',
                        f'Time: {sampling_time + to_time + train_time}s')
        logger.print_statistics(run)
    
    avg_sampling_time /= args.runs * args.epochs
    avg_to_time /= args.runs * args.epochs
    avg_train_time /= args.runs * args.epochs
    print(f'Avg_sampling_time: {avg_sampling_time}s, '
          f'Avg_to_time: {avg_to_time}s, ',
          f'Avg_train_time: {avg_train_time}s')
    
    logger.print_statistics()
    logger.save("cluster_gcn_" + str(args.batch_size))


if __name__ == "__main__":
    main()