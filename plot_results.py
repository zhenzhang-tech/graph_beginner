"""生成实验结果对比图（matplotlib），输出到 figures/ 目录。"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

models = ['GCN', 'GAT', 'GraphSAGE', 'GIN']


def grouped_bar(ax, groups, series, title, ylabel):
    x = np.arange(len(groups))
    width = 0.2
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)


# 1. 节点分类：四模型在 Cora / Computers 上
fig, ax = plt.subplots(figsize=(8, 4.5))
grouped_bar(ax, ['Cora', 'Computers'],
            {'GCN': [0.806, 0.812], 'GAT': [0.779, 0.813],
             'GraphSAGE': [0.818, 0.801], 'GIN': [0.682, 0.689]},
            'Node Classification (test accuracy)', 'Accuracy')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'task1_node_cls.png'), dpi=150)
plt.close(fig)

# 2. 链路预测：四模型在 Cora / Computers 上
fig, ax = plt.subplots(figsize=(8, 4.5))
grouped_bar(ax, ['Cora', 'Computers'],
            {'GCN': [0.823, 0.803], 'GAT': [0.905, 0.661],
             'GraphSAGE': [0.819, 0.637], 'GIN': [0.789, 0.907]},
            'Link Prediction (ROC-AUC)', 'AUC')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'task2_link_pred.png'), dpi=150)
plt.close(fig)

# 3. 池化对比（GCN/PROTEINS, GIN/MUTAG）
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(3)
width = 0.35
ax.bar(x - width / 2, [0.804, 0.777, 0.679], width, label='GCN/PROTEINS',
       color='#4C72B0')
ax.bar(x + width / 2, [0.842, 0.842, 0.579], width, label='GIN/MUTAG',
       color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(['Avg', 'Max', 'Min'])
ax.set_ylabel('Accuracy')
ax.set_title('Graph Classification: Pooling Comparison')
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'task3_pooling.png'), dpi=150)
plt.close(fig)

# 4. 知识图谱：三模型 MRR
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['TransE', 'RotatE', 'ConvE'], [0.179, 0.179, 0.032],
       color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_ylabel('MRR')
ax.set_title('Knowledge Graph Completion (FB15k-237, 30 epochs)')
ax.set_ylim(0, 0.25)
for i, v in enumerate([0.179, 0.179, 0.032]):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'task4_kge.png'), dpi=150)
plt.close(fig)

print('figures saved to', FIG_DIR)
