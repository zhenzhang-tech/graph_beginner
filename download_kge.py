"""下载 FB15k-237 知识图谱数据集（Freebase mid 格式，需自行映射数字 ID）。

用法:
    python download_kge.py
"""
import os
import urllib.request

RAW_DIR = r'D:/bysj/graph_data/FB15k-237'
BASE = ('https://ghproxy.net/https://raw.githubusercontent.com/villmow/'
        'datasets_knowledge_embedding/master/FB15k-237/')
FILES = ['train.txt', 'valid.txt', 'test.txt']

os.makedirs(RAW_DIR, exist_ok=True)


def download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as f:
        f.write(r.read())


for fname in FILES:
    dest = os.path.join(RAW_DIR, fname)
    print(f'fetch  {fname}')
    download(BASE + fname, dest)
    with open(dest) as f:
        n = sum(1 for _ in f)
    size = os.path.getsize(dest)
    print(f'  ok   {fname} ({n} lines, {size} bytes)')

print('ALL_DONE')
