# 图神经网络入门任务（graph_beginner）

基于 [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) 框架，完成图学习的四类任务。

## 任务列表

| 目录 | 任务 | 模型 |
|------|------|------|
| 任务一 | 节点分类 | GCN / GAT / GraphSAGE / GIN + 全图 vs 采样对比 |
| 任务二 | 链路预测 | 同上四模型 + 采样对比 |
| 任务三 | 图分类 | 四模型 + Avg/Max/Min 池化对比 |
| 任务四 | 知识图谱补全 | TransE / RotatE / ConvE |

## 文档

- **实验报告**：[report.md](report.md)（实验环境、方法、结果分析、问题处理）
- 各任务说明见对应目录的 `README.md`
- 运行日志见 `logs/`，结果对比图见 `figures/`

## 环境

Python 3.9 / PyTorch 2.0.0+cu118 / PyG 2.4.0，详见 [requirements.txt](requirements.txt)。

> 注：数据缓存于英文路径 `D:/bysj/graph_data`（见各任务 data 目录说明）；Flickr、ZINC 因网络受限用 Amazon、TUDataset 替代（详见报告第四节）。
