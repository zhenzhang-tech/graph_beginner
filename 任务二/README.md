# 任务二：链路预测

基于 GNN 主流模型（GCN / GAT / GraphSAGE / GIN）的链路预测，对比全图训练与采样子图训练的性能与运行时间。

## 目录结构

```
任务二/
├── data/                      # 数据集缓存（实际在 D:/bysj/graph_data）
├── code/
│   ├── link_models.py         # encoder（四模型）+ 点积 decoder
│   ├── link_prediction.py     # 全图链路预测训练脚本
│   └── link_prediction_sampling.py  # 采样链路预测训练脚本（LinkNeighborLoader）
└── README.md
```

## 环境依赖

见项目根目录 `requirements.txt`（torch 2.0.0+cu118 / torch_geometric 2.4.0 / torch_scatter / torch_sparse）。

## 数据说明

使用 Cora、Citeseer、Amazon Computers、Amazon Photo（Flickr 因数据在 Google Drive 无法下载，用 Amazon 替代，同任务一）。边划分用 `RandomLinkSplit`（train/val/test = 85%/5%/10%，训练正边 + 负采样）。

## 训练脚本

### 全图训练

```bash
cd 任务二/code
python link_prediction.py --model GCN --dataset Cora
python link_prediction.py --model GAT --dataset Computers
```

### 采样训练（LinkNeighborLoader）

```bash
python link_prediction_sampling.py --model GCN --dataset Cora --epochs 50 --batch_size 1024 --num_neighbors 5
```

主要参数：`--model`、`--dataset`、`--hidden`、`--num_layers`、`--lr`、`--epochs`；采样另有 `--batch_size`、`--num_neighbors`。

## 测试脚本

训练脚本输出 `RESULT best_val_auc=... best_test_auc=... time=...`（AUC 用 ROC-AUC）。测试即复用评估流程。

## 结果汇总（AUC，hidden=64，2 层）

### 全图训练（200 epoch）

| 模型 | Cora | Citeseer | Computers | Photo |
|------|------|----------|-----------|-------|
| GCN | 0.823 | 0.831 | 0.803 | 0.810 |
| GAT | 0.905 | 0.910 | 0.661 | 0.563 |
| GraphSAGE | 0.819 | 0.764 | 0.637 | 0.576 |
| GIN | 0.789 | 0.689 | 0.907 | 0.923 |

### 采样训练（GCN 代表，50 epoch，batch 1024，num_neighbors 5）

| 数据集 | 采样 AUC | 全图 AUC | 采样耗时 | 全图耗时 |
|--------|----------|----------|----------|----------|
| Cora | 0.864 | 0.823 | 9.25s | 3.85s |
| Citeseer | 0.821 | 0.831 | 11.76s | 2.56s |
| Computers | 0.500* | 0.803 | 2002s* | 13.55s |

> *Computers 平均度约 36，num_neighbors=5 采样严重不足，AUC 退化到随机水平（0.5）；增大邻居数则子图过大、训练极慢（且受本机并发 NLP 任务占用 GPU 影响）。这说明采样训练需按图平均度设置足够的 num_neighbors，且其收益主要在内存而非时间。

## 主要观察

1. **小图（Cora/Citeseer）采样慢于全图**：采样训练因 LinkNeighborLoader 采样 + 负采样开销，耗时高于全图（Cora 9.25s vs 3.85s），精度相近。
2. **模型与数据集交互明显**：GAT 在 Cora/Citeseer 上 AUC 最高（0.91），但在大图 Computers/Photo 全图上退化（0.66/0.56）；GIN 相反，在大图上最优（0.91/0.92）。这可能与 GAT 的注意力在大图全图训练上收敛困难、GIN 的 sum 聚合 + BatchNorm 在大图上表达力更强有关。
3. **采样训练的 epoch 数**：采样训练每个 epoch 遍历全部训练边，信息量更大，故用 50 epoch（全图用 200），二者精度可比。

## 报告要点

1. 对比全图训练与分批次训练的性能和运行时间。
2. 分析不同模型在链路预测任务上的表现差异。
