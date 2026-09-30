# 任务四：知识图谱补全

基于知识图谱补全（KGE）模型 TransE / RotatE / ConvE，在 FB15k-237 上做链接预测，评估 filtered MRR / Hits@K。

## 目录结构

```
任务四/
├── data/                      # 数据集（实际在 D:/bysj/graph_data/FB15k-237）
├── code/
│   ├── kge_models.py          # TransE / RotatE / ConvE 模型定义
│   └── train_kge.py           # 训练 + 负采样 + filtered MRR/Hits@K 评估
└── README.md
```

## 数据说明

使用 **FB15k-237**（14,541 实体，237 关系，train 272,115 / valid 17,535 / test 20,466 三元组）。原始数据为 Freebase mid 格式，`train_kge.py` 内将其映射为数字 ID。数据通过 GitHub 镜像下载（`download_kge.py`）。

## 训练脚本

```bash
cd 任务四/code
python train_kge.py --model TransE --epochs 30 --lr 0.01
python train_kge.py --model RotatE --epochs 30 --lr 0.01
python train_kge.py --model ConvE --epochs 30 --lr 0.01
```

参数：`--model`（TransE/RotatE/ConvE）、`--dim`、`--lr`、`--epochs`、`--batch_size`、`--neg_samples`、`--margin`、`--eval_subset`（评估子集大小，默认 2000）。

## 测试脚本

训练脚本每个 epoch 用负采样训练（corrupt 头/尾实体），每 5 epoch 在 valid 子集上评估 filtered MRR/Hits@K，最终输出 `RESULT test MRR=... Hits@1/3/10=...`。

## 模型原理

| 模型 | 核心思想 | 打分函数 | 特点 |
|------|----------|----------|------|
| TransE | 翻译假设 | −‖h + r − t‖ | 简单高效，难建模一对多/对称关系 |
| RotatE | 复数空间旋转 | −‖h∘r − t‖ | 可建模对称/反对称/逆/组合关系 |
| ConvE | 2D 卷积编码 | f([h;r]∗ω)·t | 表达力强，计算更贵 |

- **TransE**：实体和关系在同一向量空间，头实体 + 关系 ≈ 尾实体。
- **RotatE**：实体为复数向量，关系为旋转相位，头实体旋转后接近尾实体；旋转的群结构使它能建模多种关系模式。
- **ConvE**：头实体与关系拼接成 2D 矩阵，经卷积得到联合表示，再与所有尾实体点积打分。

## 结果（FB15k-237，30 epoch，lr=0.01，margin loss + 负采样）

| 模型 | MRR | Hits@1 | Hits@3 | Hits@10 |
|------|-----|--------|--------|---------|
| TransE | 0.179 | 0.110 | 0.192 | 0.319 |
| RotatE | 0.179 | 0.112 | 0.194 | 0.313 |
| ConvE | 0.032 | 0.015 | 0.026 | 0.059 |

> 说明：
> 1. TransE / RotatE 用 margin loss + 负采样可正常收敛，二者 MRR 接近。
> 2. ConvE 用同样的 margin loss 收敛不佳——ConvE 原始论文采用 **BCE + 1-N scoring**（对每个头实体+关系，一次性对所有尾实体打分）训练，margin loss 的负采样方式与其点积打分不匹配。
> 3. 30 epoch 下 TransE/RotatE 尚未完全收敛（FB15k-237 上完整训练约 MRR 0.28 / 0.34），仅用于演示三模型的实现与相对差异。

## 报告要点

1. 理解 TransE / RotatE / ConvE 的原理差异。
2. 数据集训练/验证/测试划分（train/valid/test.txt）。
