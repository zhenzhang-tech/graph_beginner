"""下载 Amazon Computers/Photo 数据集（替代 Flickr），含完整性验证。

Flickr 的 feats.npy 约 178MB，超过 GitHub 100MB 单文件限制，只能放 Google Drive，
国内无法下载。Amazon 数据集规模介于 Cora 和 Flickr 之间，可通过 GitHub 镜像下载。

用法:
    python download_amazon.py
"""
import os
import urllib.request

import numpy as np

RAW_ROOT = r'D:/bysj/graph_data'
BASE = 'https://ghproxy.net/https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/'
FILES = {
    'Computers': 'amazon_electronics_computers.npz',
    'Photo': 'amazon_electronics_photo.npz',
}


def is_valid(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        np.load(path)
        return True
    except Exception:
        return False


for name, fname in FILES.items():
    raw_dir = os.path.join(RAW_ROOT, name, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, fname)

    if is_valid(dest):
        print(f'skip   {name} (valid, {os.path.getsize(dest)} bytes)')
        continue

    if os.path.exists(dest):
        os.remove(dest)
        print(f'removed corrupt {name}')

    url = BASE + fname
    print(f'fetch  {name} ({fname})')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, 'wb') as f:
        f.write(resp.read())

    if is_valid(dest):
        print(f'  ok   {name} ({os.path.getsize(dest)} bytes)')
    else:
        print(f'  FAILED {name}')

print('ALL_DONE')
