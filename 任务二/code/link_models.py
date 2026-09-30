"""链路预测 encoder（四模型，输出节点 embedding）与 decoder（点积）。"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2,
                 dropout=0.5, heads=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
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


class SAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GINEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
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


class DotDecoder(torch.nn.Module):
    def forward(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
