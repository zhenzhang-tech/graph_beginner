"""图分类训练脚本：四种 GNN 编码器 + 三种池化（mean/max/min）。

用法:
    python graph_classification.py --model GCN --dataset PROTEINS --pool mean
    python graph_classification.py --model GIN --dataset ENZYMES --pool max
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GCNConv, GATConv, SAGEConv, GINConv,
                                global_mean_pool, global_max_pool)
from torch_geometric.utils import scatter

DATA_ROOT = r'D:/bysj/graph_data'


def global_min_pool(x, batch, size=None):
    return scatter(x, batch, dim=0, dim_size=size, reduce='min')


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            in_channels = hidden_channels
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


ENCODERS = {
    'GCN': GCNEncoder,
    'GAT': GATEncoder,
    'GraphSAGE': SAGEEncoder,
    'GIN': GINEncoder,
}


class GraphClassifier(nn.Module):
    def __init__(self, encoder, hidden_channels, num_classes, pool='mean'):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        z = self.encoder(x, edge_index)
        if self.pool == 'mean':
            g = global_mean_pool(z, batch)
        elif self.pool == 'max':
            g = global_max_pool(z, batch)
        elif self.pool == 'min':
            g = global_min_pool(z, batch, size=batch.max().item() + 1)
        else:
            raise ValueError(f'Unknown pool: {self.pool}')
        return self.classifier(g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCN', choices=list(ENCODERS))
    parser.add_argument('--dataset', default='PROTEINS',
                        choices=['PROTEINS', 'ENZYMES', 'MUTAG'])
    parser.add_argument('--pool', default='mean', choices=['mean', 'max', 'min'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TUDataset(root=DATA_ROOT, name=args.dataset)

    # 固定随机划分 8:1:1
    perm = torch.randperm(len(dataset)).tolist()
    n = len(dataset)
    train_idx = perm[:int(0.8 * n)]
    val_idx = perm[int(0.8 * n):int(0.9 * n)]
    test_idx = perm[int(0.9 * n):]

    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size)

    model = GraphClassifier(
        encoder=ENCODERS[args.model](dataset.num_features, args.hidden,
                                     args.num_layers, args.dropout),
        hidden_channels=args.hidden,
        num_classes=dataset.num_classes,
        pool=args.pool,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=5e-4)

    def train():
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_idx)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        correct = total = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
        return correct / total

    print(f'model={args.model} dataset={args.dataset} pool={args.pool} '
          f'hidden={args.hidden} layers={args.num_layers} device={device}')

    best_val = 0.0
    best_test = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_acc = evaluate(val_loader)
        test_acc = evaluate(test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | loss {loss:.4f} | val {val_acc:.4f} '
                  f'| test {test_acc:.4f}')
    elapsed = time.time() - start
    print(f'RESULT best_val={best_val:.4f} best_test={best_test:.4f} '
          f'time={elapsed:.2f}s')


if __name__ == '__main__':
    main()
