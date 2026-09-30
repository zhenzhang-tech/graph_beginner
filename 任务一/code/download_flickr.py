"""下载 Flickr 数据集（需能访问 Google Drive 的网络环境，如挂代理）。

Flickr 的 4 个原始文件托管在 Google Drive，国内网络无法直连。
请在能访问 Google Drive 的环境下运行本脚本，或手动下载后放入
D:/bysj/graph_data/Flickr/raw/ 目录。

用法:
    pip install gdown
    python download_flickr.py
"""
import os
import gdown

RAW_DIR = r'D:/bysj/graph_data/Flickr/raw'
FILES = {
    'adj_full.npz': '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy',
    'feats.npy': '1join-XdvX3anJU_MLVtick7MgeAQiWIZ',
    'class_map.json': '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9',
    'role.json': '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7',
}

os.makedirs(RAW_DIR, exist_ok=True)
for fname, file_id in FILES.items():
    dest = os.path.join(RAW_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'skip   {fname}')
        continue
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f'fetch  {fname}')
    gdown.download(url, dest, quiet=False)
    print(f'  ok   {fname}')

print('ALL_DONE')
