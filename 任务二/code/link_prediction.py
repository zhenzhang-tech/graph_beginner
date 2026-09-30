"""链路预测训练脚本（全图训练，负采样）。

用法:
    python link_prediction.py --model GCN --dataset Cora
"""
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score

from link_models import ENCODERS, DotDecoder

DATA_ROOT = r'D:/bysj/graph_data'


def load_dataset(name):
    if name in ('Cora', 'Citeseer', 'PubMed'):
        data = Planetoid(root=DATA_ROOT, name=name)[0]
    elif name in ('Computers', 'Photo'):
        data = Amazon(root=DATA_ROOT, name=name)[0]
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCN', choices=list(ENCODERS))
    parser.add_argument('--dataset', default='Cora',
                        choices=['Cora', 'Citeseer', 'Computers', 'Photo'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(args.dataset)

    transform = RandomLinkSplit(
        num_val=0.05, num_test=0.1, is_undirected=True,
        add_negative_train_samples=True, neg_sampling_ratio=1.0)
    train_data, val_data, test_data = transform(data)

    encoder = ENCODERS[args.model](
        in_channels=data.num_features, hidden_channels=args.hidden,
        num_layers=args.num_layers, dropout=args.dropout).to(device)
    decoder = DotDecoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    def train():
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        z = encoder(train_data.x, train_data.edge_index)
        out = decoder(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(split):
        encoder.eval()
        decoder.eval()
        z = encoder(split.x, split.edge_index)
        out = decoder(z, split.edge_label_index)
        return roc_auc_score(split.edge_label.cpu().numpy(),
                             out.cpu().numpy())

    print(f'model={args.model} dataset={args.dataset} hidden={args.hidden} '
          f'layers={args.num_layers} device={device}')

    best_val = 0.0
    best_test = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_auc = evaluate(val_data)
        test_auc = evaluate(test_data)
        if val_auc > best_val:
            best_val = val_auc
            best_test = test_auc
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | loss {loss:.4f} | val {val_auc:.4f} '
                  f'| test {test_auc:.4f}')
    elapsed = time.time() - start
    print(f'RESULT best_val_auc={best_val:.4f} best_test_auc={best_test:.4f} '
          f'time={elapsed:.2f}s')


if __name__ == '__main__':
    main()
