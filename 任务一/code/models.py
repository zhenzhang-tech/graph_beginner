"""GNN 节点分类模型：GCN / GAT / GraphSAGE / GIN。

每个模型都支持 num_layers 和 dropout 两个超参数，
forward(x, edge_index) -> logits。
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * heads, out_channels,
                                  heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, out_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(torch.nn.BatchNorm1d(out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


MODELS = {
    'GCN': GCN,
    'GAT': GAT,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
}
