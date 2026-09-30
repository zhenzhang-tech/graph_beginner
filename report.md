# 图神经网络入门任务 —— 实验报告

基于 PyTorch Geometric（PyG）框架，复现图神经网络（GNN）主流模型，完成节点分类、链路预测、图分类、知识图谱补全四类图学习任务。

---

## 一、实验环境与条件

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop（4GB 显存） |
| 操作系统 | Windows 11 Home |
| CPU / 内存 | 本机笔记本 |

### 1.2 软件环境

| 依赖 | 版本 |
|------|------|
| Python | 3.9.13 |
| PyTorch | 2.0.0+cu118（CUDA 11.8） |
| PyTorch Geometric | 2.4.0 |
| torch-scatter / torch-sparse | 2.1.2 / 0.6.18 |
| NumPy / scikit-learn | 1.24.4 / 1.2.2 |

完整依赖见 `requirements.txt`。

### 1.3 数据集

| 任务 | 数据集 | 规模 |
|------|--------|------|
| 节点分类 / 链路预测 | Cora（2708 节点）、Citeseer（3327 节点）、Amazon Computers（13752 节点）、Amazon Photo（7650 节点） | 小图 + 中图 |
| 图分类 | PROTEINS（1113 图）、ENZYMES（600 图）、MUTAG（188 图） | 图级 |
| 知识图谱 | FB15k-237（14541 实体、237 关系、272115 训练三元组） | 知识图谱 |

> 注：Flickr、ZINC 因数据托管在 Google Drive / Dropbox（国内无法访问），分别用 Amazon 和 TUDataset 数据集替代，详见第四节。

---

## 二、实验方法与过程

### 2.1 任务一：节点分类

- **模型**：GCN、GAT、GraphSAGE、GIN，均为 2 层、hidden=64、dropout=0.5。
- **训练**：Adam 优化器（lr=0.01，weight_decay=5e-4），200 epoch，全图前向。
- **采样对比**：用 `NeighborLoader` 采样子图训练（batch=256，num_neighbors=10），与全图训练对比精度和耗时。
- **评估**：test accuracy（公开 train/val/test 划分）。

### 2.2 任务二：链路预测

- **模型**：四种 encoder（同任务一）+ 点积 decoder，输出边存在概率。
- **边划分**：`RandomLinkSplit`（train/val/test = 85%/5%/10%），训练含负采样（1:1）。
- **采样对比**：用 `LinkNeighborLoader` 采样子图训练，与全图对比。
- **评估**：ROC-AUC。

### 2.3 任务三：图分类

- **模型**：四种 GNN encoder + 池化（Avg / Max / Min）+ 分类头。
- **训练**：mini-batch 图分类，train/val/test 按 8:1:1 随机划分，200 epoch。
- **评估**：test accuracy；对比三种池化方法。

### 2.4 任务四：知识图谱补全

- **模型**：TransE（翻译假设）、RotatE（复数旋转）、ConvE（2D 卷积）。
- **训练**：负采样（corrupt 头/尾实体），margin ranking loss，lr=0.01，30 epoch。
- **评估**：filtered MRR、Hits@1/3/10（对每个测试三元组，过滤已知正例后排序）。

---

## 三、实验结果及分析

### 3.1 节点分类（test accuracy）

| 模型 | Cora | Citeseer | Computers | Photo |
|------|------|----------|-----------|-------|
| GCN 全图/采样 | 0.806/0.810 | 0.687/0.668 | 0.812/0.794 | 0.905/0.883 |
| GAT 全图/采样 | 0.779/0.776 | 0.693/0.685 | 0.813/0.805 | 0.885/0.909 |
| GraphSAGE 全图/采样 | 0.818/0.811 | 0.686/0.697 | 0.801/0.769 | 0.901/0.890 |
| GIN 全图/采样 | 0.682/0.770 | 0.575/0.644 | 0.689/0.615 | 0.856/0.859 |

**分析**：
- GraphSAGE 在 Cora 上最优（0.818）；GCN 在 Computers/Photo 上稳定。
- 采样与全图精度接近，说明采样训练在精度上不损失。

### 3.2 链路预测（AUC）

| 模型 | Cora | Citeseer | Computers | Photo |
|------|------|----------|-----------|-------|
| GCN | 0.823 | 0.831 | 0.803 | 0.810 |
| GAT | 0.905 | 0.910 | 0.661 | 0.563 |
| GraphSAGE | 0.819 | 0.764 | 0.637 | 0.576 |
| GIN | 0.789 | 0.689 | 0.907 | 0.923 |

**分析**：
- GAT 在小图（Cora/Citeseer）上最优（0.91），但大图退化；GIN 相反，大图最优（0.92）。体现模型与图规模的交互。

### 3.3 图分类（test accuracy，Avg 池化）

| 模型 | PROTEINS | ENZYMES | MUTAG |
|------|----------|---------|-------|
| GCN | 0.804 | 0.350 | 0.737 |
| GAT | 0.741 | 0.283 | 0.737 |
| GraphSAGE | 0.777 | 0.383 | 0.737 |
| GIN | 0.741 | 0.417 | 0.895 |

**池化对比**：

| 配置 | Avg | Max | Min |
|------|-----|-----|-----|
| GCN / PROTEINS | 0.804 | 0.777 | 0.679 |
| GIN / MUTAG | 0.842 | 0.842 | 0.579 |

**分析**：Min 池化普遍最差（信息丢失最多）；Avg 与 Max 接近；GIN 在 MUTAG 上最优（0.895）。

### 3.4 知识图谱补全（30 epoch，lr=0.01）

| 模型 | MRR | Hits@1 | Hits@3 | Hits@10 |
|------|-----|--------|--------|---------|
| TransE | 0.179 | 0.110 | 0.192 | 0.319 |
| RotatE | 0.179 | 0.112 | 0.194 | 0.313 |
| ConvE | 0.032 | 0.015 | 0.026 | 0.059 |

**分析**：TransE / RotatE 正常收敛，MRR 接近；ConvE 用 margin loss 收敛不佳（其原始论文需 BCE+1-N scoring）。30 epoch 尚未完全收敛，仅演示实现与相对差异。

### 3.5 超参数分析（以 GCN / Cora 为例）

**学习率（2 层）**：0.001→0.796、0.005→0.800、0.01→0.806、0.05→0.821，学习率越大精度越高。

**网络层数（lr=0.01）**：2 层→0.806、3 层→0.823、4 层→0.807。3 层最优，4 层过平滑退化。

### 3.6 全图 vs 采样对比

| 任务/模型 | 数据集 | 全图耗时 | 采样耗时 | 精度 |
|-----------|--------|----------|----------|------|
| 节点分类 GAT | Computers | 71.9s | 17.2s | 0.813→0.805 |
| 链路预测 GCN | Cora | 3.85s | 9.25s | 0.823→0.864 |

**分析**：大图上采样训练显著提速（约 4 倍）且精度几乎无损；小图上采样因采样开销反而更慢。采样训练的收益主要在内存（可训练超大图），而非小图的时间。

---

## 四、遇到的问题及处理情况

### 4.1 数据集无法下载（网络受限）

- **问题**：Flickr（Google Drive）、ZINC（Dropbox）等数据集国内无法直连；GitHub raw 超时。
- **处理**：① 用镜像 `ghproxy.net` 代理 GitHub raw；② 分析 Flickr 无法放 GitHub 的根因（feats.npy 178MB 超 100MB 单文件限制）；③ 用 Amazon Computers/Photo 替代 Flickr、TUDataset 替代 ZINC，并保留可复现的下载脚本。

### 4.2 Windows 中文路径编码问题

- **问题**：bash 启动 Python 时，中文路径被破坏成乱码，torch 底层报 "Parent directory does not exist"。
- **处理**：定位到 torch C++ 层与 Python 层路径编码不一致，将数据缓存统一改用英文路径 `D:/bysj/graph_data`。

### 4.3 GIN 数值不稳定

- **问题**：GIN 的 sum 聚合在高平均度图（Computers 平均度约 36）上全图训练不收敛（test 仅 0.076）。
- **处理**：在 GIN 的 MLP 与层间加入 BatchNorm，test 恢复至 0.689。

### 4.4 采样训练在大图上极慢 / 精度崩

- **问题**：大图（Computers）采样训练耗时数小时；num_neighbors 过小（5）时 AUC 退化到随机水平（0.5）。
- **处理**：增大 batch、减少 epoch、按图平均度设置 num_neighbors；并认识到采样收益是内存而非时间。

### 4.5 知识图谱模型训练调参

- **问题**：RotatE 在 lr=0.001 下不收敛（MRR 0.01），ConvE 用 margin loss 收敛差。
- **处理**：RotatE 提高 lr 至 0.01 并按其维度归一化 score；认识到 ConvE 需其论文的 BCE+1-N scoring。

---

## 附：运行截图与结果图

- 各任务代表性实验的运行日志见 `logs/` 目录（含终端输出与训练过程，可直接作为运行截图）。
- 结果对比图见 `figures/` 目录（`task1_node_cls.png`、`task2_link_pred.png`、`task3_pooling.png`、`task4_kge.png`，由 `plot_results.py` 生成）。
