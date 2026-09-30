# 任务一：节点分类

基于 GNN 主流模型（GCN / GAT / GraphSAGE / GIN）的节点分类，对比全图训练与采样子图训练的性能与运行时间。

## 目录结构

```
任务一/
├── data/                      # 数据集缓存（实际在 D:/bysj/graph_data）
├── code/
│   ├── models.py               # 四种模型定义
│   ├── train_node_cls.py       # 全图训练脚本
│   ├── train_node_cls_sampling.py  # 采样训练脚本（NeighborLoader）
│   ├── train_gcn.py            # 最小 GCN demo（环境验证）
│   ├── download_flickr.py      # Flickr 下载脚本（需代理，见下方说明）
│   └── download_amazon.py      # Amazon 下载脚本（项目根目录另有 download_planetoid.py）
└── README.md
```

## 环境依赖

见项目根目录 `requirements.txt`。关键依赖：

- torch 2.0.0+cu118（CUDA 版）
- torch_geometric 2.4.0
- torch_scatter 2.1.2 / torch_sparse 0.6.18（采样训练必需，从 data.pyg.org 安装）

## 数据说明

- **Cora / Citeseer**：通过 GitHub 镜像下载（`download_planetoid.py`）。
- **Amazon Computers / Photo**：通过 GitHub 镜像下载（`download_amazon.py`）。
- **关于 Flickr**：原任务要求使用 Flickr，但其 `feats.npy` 约 178MB，超过 GitHub 100MB 单文件限制，只能托管在 Google Drive，国内网络无法直连。故用 **Amazon Computers（13,752 节点）/ Photo（7,650 节点）** 替代，二者规模比 Cora/Citeseer 大一个量级，能更好地体现采样训练在大图上的优势。

> 注：数据缓存统一用英文路径 `D:/bysj/graph_data`，规避 Windows 下中文路径导致的 torch 底层编码问题。

## 训练脚本

### 全图训练

```bash
cd 任务一/code
python train_node_cls.py --model GCN --dataset Cora
python train_node_cls.py --model GAT --dataset Computers --num_layers 3
python train_node_cls.py --model GraphSAGE --dataset Photo
python train_node_cls.py --model GIN --dataset Citeseer
```

主要参数：`--model`（GCN/GAT/GraphSAGE/GIN）、`--dataset`（Cora/Citeseer/Computers/Photo）、`--hidden`、`--num_layers`、`--lr`、`--dropout`、`--epochs`。

### 采样训练（NeighborLoader）

```bash
python train_node_cls_sampling.py --model GCN --dataset Cora
python train_node_cls_sampling.py --model GraphSAGE --dataset Computers --batch_size 512 --num_neighbors 25
```

采样参数：`--batch_size`、`--num_neighbors`（每层采样邻居数，自动按层数复制）。

## 测试脚本

训练脚本每个 epoch 在 train/val/test 上评估，最终输出 `RESULT best_val=... best_test=... time=...`（time 为训练总耗时）。测试即复用该评估流程。

## 结果汇总（hidden=64，2 层，lr=0.01）

test accuracy：

| 模型 | Cora 全图/采样 | Citeseer 全图/采样 | Computers 全图/采样 | Photo 全图/采样 |
|------|------|------|------|------|
| GCN | 0.806 / 0.810 | 0.687 / 0.668 | 0.812 / 0.794 | 0.905 / 0.883 |
| GAT | 0.779 / 0.776 | 0.693 / 0.685 | 0.813 / 0.805 | 0.885 / 0.909 |
| GraphSAGE | 0.818 / 0.811 | 0.686 / 0.697 | 0.801 / 0.769 | 0.901 / 0.890 |
| GIN | 0.682 / 0.770 | 0.575 / 0.644 | 0.689 / 0.615 | 0.856 / 0.859 |

训练耗时（秒）：

| 模型 | Cora 全图/采样 | Computers 全图/采样 |
|------|------|------|
| GCN | 1.94 / 2.48 | 5.00 / 6.08 |
| GAT | 3.69 / 3.64 | 71.85 / 17.16 |
| GraphSAGE | 2.04 / 2.28 | 21.08 / 14.10 |
| GIN | 2.18 / 2.03 | 21.95 / 15.65 |

## 主要观察

1. **小图（Cora/Citeseer）**：全图与采样精度接近，采样因额外开销略慢。
2. **大图（Computers）**：GAT 全图训练 71.9s，采样仅 17.2s，**采样提速约 4 倍**，且精度损失很小（0.813 → 0.805）。
3. **GIN 的数值稳定性**：GIN 的 sum 聚合在高平均度图（Computers 平均度约 36）上全图训练易不收敛（无 BatchNorm 时 test 仅 0.076），加入 BatchNorm 后恢复至 0.689。这体现了归一化对 GIN 的必要性。

## 报告要点

1. 分析不同学习率、网络层数对性能的影响（`--lr` / `--num_layers`）。
2. 对比全图训练与分批次训练的性能和运行时间（`train_node_cls.py` vs `train_node_cls_sampling.py`）。
