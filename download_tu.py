"""下载 TUDataset 图分类数据集（PROTEINS/ENZYMES/MUTAG）。

TUDataset 源需要 User-Agent，PyG 内置下载不带 UA 会 403，故手动下载解压。
"""
import os
import zipfile
import urllib.request

BASE = 'https://www.chrsmrrs.com/graphkerneldatasets'
RAW_ROOT = r'D:/bysj/graph_data'
DATASETS = ['PROTEINS', 'ENZYMES', 'MUTAG']


def download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, 'wb') as f:
        f.write(resp.read())


for name in DATASETS:
    raw_dir = os.path.join(RAW_ROOT, name, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    # 已解压则跳过
    if os.path.exists(os.path.join(raw_dir, f'{name}_A.txt')):
        print(f'skip   {name}')
        continue

    zip_path = os.path.join(RAW_ROOT, name, f'{name}.zip')
    url = f'{BASE}/{name}.zip'
    print(f'fetch  {name}')
    download(url, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            basename = os.path.basename(member)
            if not basename:
                continue
            with z.open(member) as src, \
                    open(os.path.join(raw_dir, basename), 'wb') as dst:
                dst.write(src.read())
    os.remove(zip_path)
    n = len(os.listdir(raw_dir))
    print(f'  ok   {name} ({n} files)')

print('ALL_DONE')
