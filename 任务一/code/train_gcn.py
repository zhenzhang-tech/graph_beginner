"""Minimal GCN demo on Cora to verify environment (M0).

Run:
    python train_gcn.py
"""
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# 数据缓存根目录：使用英文路径，避免中文路径在 Windows 下的编码问题
DATA_ROOT = r'D:/bysj/graph_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

dataset = Planetoid(root=DATA_ROOT, name='Cora')
data = dataset[0].to(device)
print(f'Cora: {data.num_nodes} nodes, {data.num_edges} edges, '
      f'{dataset.num_features} feats, {dataset.num_classes} classes')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(dataset.num_features, 64, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = evaluate()
        print(f'Epoch {epoch:03d} | loss {loss:.4f} | '
              f'train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}')

print('DONE')
