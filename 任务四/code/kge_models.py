"""知识图谱补全模型：TransE / RotatE / ConvE。

每个模型实现 score(h, r, t)，返回三元组得分（越大越可能为正例）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    """头实体 + 关系 接近 尾实体：score = -||h + r - t||。"""

    def __init__(self, num_entities, num_relations, dim=100, p_norm=2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.p_norm = p_norm
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.uniform_(self.rel_emb.weight, -1.0 / math.sqrt(dim),
                         1.0 / math.sqrt(dim))

    def score(self, h, r, t):
        h_emb = F.normalize(self.entity_emb(h), p=2, dim=-1)
        r_emb = self.rel_emb(r)
        t_emb = F.normalize(self.entity_emb(t), p=2, dim=-1)
        return -torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=-1)


class RotatE(nn.Module):
    """关系表示为复数空间中的旋转：score = -||h ∘ r - t||^2。"""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.dim = dim
        # 实体 embedding 存实部 + 虚部，共 2*dim
        self.entity_emb = nn.Embedding(num_entities, dim * 2)
        # 关系 embedding 存相位（角度）
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.rel_emb.weight, -math.pi, math.pi)

    def score(self, h, r, t):
        h = self.entity_emb(h)
        t = self.entity_emb(t)
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        # 实体归一化到单位圆（RotatE 标准做法）
        h_norm = torch.sqrt(h_re ** 2 + h_im ** 2 + 1e-12)
        t_norm = torch.sqrt(t_re ** 2 + t_im ** 2 + 1e-12)
        h_re, h_im = h_re / h_norm, h_im / h_norm
        t_re, t_im = t_re / t_norm, t_im / t_norm
        phase = self.rel_emb(r)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        return -torch.sqrt(torch.sum((hr_re - t_re) ** 2 + (hr_im - t_im) ** 2,
                                     dim=-1)) / math.sqrt(self.dim)


class ConvE(nn.Module):
    """2D 卷积 + 全连接打分（Dettmers et al. 2018）。"""

    def __init__(self, num_entities, num_relations, dim=100, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.conv1 = nn.Conv2d(1, 32, (2, 3), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        fc_in = 32 * 1 * (dim - 2)
        self.fc = nn.Linear(fc_in, dim)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _encode(self, h, r):
        h_emb = self.entity_emb(h).view(-1, 1, self.dim)
        r_emb = self.rel_emb(r).view(-1, 1, self.dim)
        stacked = torch.cat([h_emb, r_emb], dim=1).unsqueeze(1)  # (B,1,2,dim)
        x = F.relu(self.bn1(self.conv1(stacked)))
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)

    def score(self, h, r, t):
        x = self._encode(h, r)
        t_emb = self.entity_emb(t)
        return torch.sum(x * t_emb, dim=-1)


MODELS = {
    'TransE': TransE,
    'RotatE': RotatE,
    'ConvE': ConvE,
}
