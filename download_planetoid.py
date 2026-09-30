"""Download Cora / Citeseer raw files via a GitHub mirror.

数据缓存到英文路径 D:/bysj/graph_data，避免中文路径在 Windows 下的编码问题。
"""
import os
import urllib.request

DATA_DIR = r'D:/bysj/graph_data'
BASE = 'https://ghproxy.net/https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/'
NAMES = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
DATASETS = ['cora', 'citeseer']


def download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, 'wb') as f:
        f.write(resp.read())


for ds in DATASETS:
    raw_dir = os.path.join(DATA_DIR, ds.capitalize(), 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    for n in NAMES:
        fname = f'ind.{ds}.{n}'
        dest = os.path.join(raw_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f'skip   {fname}')
            continue
        url = BASE + fname
        print(f'fetch  {fname}')
        download(url, dest)
        print(f'  ok   {fname} ({os.path.getsize(dest)} bytes)')

print('ALL_DONE')
