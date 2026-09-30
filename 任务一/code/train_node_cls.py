"""节点分类训练脚本（全图训练）。

用法:
    python train_node_cls.py --model GCN --dataset Cora
    python train_node_cls.py --model GAT --dataset Citeseer --num_layers 3
"""
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon

from models import MODELS

DATA_ROOT = r'D:/bysj/graph_data'


def add_masks(data, train_per_class=20, val_per_class=30, seed=0):
    """为 gnn-benchmark 数据集（Amazon）生成 train/val/test 划分。"""
    import numpy as np
    num_classes = int(data.y.max()) + 1
    rng = np.random.RandomState(seed)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1).numpy()
        perm = rng.permutation(idx)
        train_mask[perm[:train_per_class]] = True
        val_mask[perm[train_per_class:train_per_class + val_per_class]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = ~(train_mask | val_mask)
    return data


def load_dataset(name):
    if name in ('Cora', 'Citeseer', 'PubMed'):
        dataset = Planetoid(root=DATA_ROOT, name=name)
        data = dataset[0]
    elif name in ('Computers', 'Photo'):
        dataset = Amazon(root=DATA_ROOT, name=name)
        data = add_masks(dataset[0])
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return dataset, data


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCN', choices=list(MODELS))
    parser.add_argument('--dataset', default='Cora',
                        choices=['Cora', 'Citeseer', 'Computers', 'Photo'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, data = load_dataset(args.dataset)
    data = data.to(device)

    model = MODELS[args.model](
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    print(f'model={args.model} dataset={args.dataset} hidden={args.hidden} '
          f'layers={args.num_layers} lr={args.lr} device={device}')

    best_val = 0.0
    best_test = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = evaluate(model, data)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | loss {loss:.4f} | train {train_acc:.4f} '
                  f'| val {val_acc:.4f} | test {test_acc:.4f}')
    elapsed = time.time() - start
    print(f'RESULT best_val={best_val:.4f} best_test={best_test:.4f} '
          f'time={elapsed:.2f}s')


if __name__ == '__main__':
    main()
