# 任务三：图分类

基于 GNN 主流模型（GCN / GAT / GraphSAGE / GIN）的图分类，分析不同池化方法（Avg / Max / Min）对性能的影响。

## 目录结构

```
任务三/
├── data/                      # 数据集缓存（实际在 D:/bysj/graph_data）
├── code/
│   └── graph_classification.py  # 图分类训练脚本（四模型 + 三池化）
└── README.md
```

## 环境依赖

见项目根目录 `requirements.txt`（torch 2.0.0+cu118 / torch_geometric 2.4.0 / torch_scatter / torch_sparse）。

## 数据说明

- **TUDataset**：使用 PROTEINS（1113 图，2 类）、ENZYMES（600 图，6 类）、MUTAG（188 图，2 类）。源 `chrsmrrs.com` 需 User-Agent，故用 `download_tu.py`（项目根目录）手动下载解压。
- **关于 ZINC**：原任务要求使用 ZINC，但其 `molecules.zip` 托管在 Dropbox（国内被墙），且需 RDKit 处理 SMILES，故改用 TUDataset 的 3 个数据集完成图分类。

> 数据缓存统一用英文路径 `D:/bysj/graph_data`。

## 训练脚本

```bash
cd 任务三/code
python graph_classification.py --model GCN --dataset PROTEINS --pool mean
python graph_classification.py --model GIN --dataset MUTAG --pool max
python graph_classification.py --model GraphSAGE --dataset ENZYMES --pool min
```

主要参数：`--model`（GCN/GAT/GraphSAGE/GIN）、`--dataset`（PROTEINS/ENZYMES/MUTAG）、`--pool`（mean/max/min）、`--hidden`、`--num_layers`、`--epochs`、`--batch_size`。

## 测试脚本

训练脚本输出 `RESULT best_val=... best_test=... time=...`，测试即复用评估流程（train/val/test 按 8:1:1 固定随机划分）。

## 结果汇总

### 四模型对比（mean 池化，hidden=64，2 层）

test accuracy：

| 模型 | PROTEINS | ENZYMES | MUTAG |
|------|----------|---------|-------|
| GCN | 0.804 | 0.350 | 0.737 |
| GAT | 0.741 | 0.283 | 0.737 |
| GraphSAGE | 0.777 | 0.383 | 0.737 |
| GIN | 0.741 | 0.417 | 0.895 |

### 池化方法对比

test accuracy：

| 配置 | Avg (mean) | Max | Min |
|------|-----------|-----|-----|
| GCN / PROTEINS | 0.804 | 0.777 | 0.679 |
| GCN / MUTAG | 0.737 | 0.684 | 0.579 |
| GIN / PROTEINS | 0.750 | 0.786 | 0.625 |
| GIN / MUTAG | 0.842 | 0.842 | 0.579 |

## 主要观察

1. **Min 池化普遍最差**：min 聚合只保留每个特征维度的最小值，丢失的信息最多，精度显著低于 Avg/Max。
2. **Avg 与 Max 接近**：多数情况下 mean 与 max 精度相当，二者均能有效聚合节点信息。
3. **GIN 在 MUTAG 上最优**（0.895），得益于其 sum 聚合 + BatchNorm 的表达能力；ENZYMES 为 6 分类且图较小，各模型精度均偏低（0.28~0.42）。

## 报告要点

1. 分析不同池化方法（Avg/Max/Min）对图分类性能的影响。
2. 对比四种模型在不同数据集上的表现。
