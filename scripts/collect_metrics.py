#!/usr/bin/env python
import os
import re
import csv
from pathlib import Path

# 扫描根目录（以当前脚本所在目录为相对基准）
ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / 'experiments' / 'train'
OUT_CSV = TRAIN_DIR / 'metrics_summary.csv'

# 匹配示例：
# Epoch 7 loss 0.8605 acc@1 0.7472 precision@1 0.7680 recall@1 0.7472 f1@1 0.7458
LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+"
    r"loss\s+(?P<loss>[0-9.]+)\s+"
    r"acc@1\s+(?P<acc>[0-9.]+)\s+"
    r"precision@1\s+(?P<precision>[0-9.]+)\s+"
    r"recall@1\s+(?P<recall>[0-9.]+)\s+"
    r"f1@1\s+(?P<f1>[0-9.]+)"
)

# 允许的模型目录（排除 resnext）
EXCLUDE_MODEL = {'resnext'}

rows = []

if TRAIN_DIR.exists():
    for root, dirs, files in os.walk(TRAIN_DIR):
        # 路径中若包含 resnext 则跳过
        parts = Path(root).parts
        if any(part in EXCLUDE_MODEL for part in parts):
            continue
        if 'log.txt' in files:
            log_path = Path(root) / 'log.txt'
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        m = LINE_RE.search(line)
                        if m:
                            # 解析模型与实验名（train/<model>/<name>/...）
                            # 例如 experiments/train/cabbage6/densenet/exp00/log.txt
                            rel = log_path.relative_to(TRAIN_DIR)
                            parts_rel = rel.parts  # ('cabbage6','densenet','exp00','log.txt')
                            dataset_or_group = parts_rel[0] if len(parts_rel) > 0 else ''
                            model_name = parts_rel[1] if len(parts_rel) > 1 else ''
                            exp_name = parts_rel[2] if len(parts_rel) > 2 else ''

                            rows.append({
                                'dataset_or_group': dataset_or_group,
                                'model': model_name,
                                'exp': exp_name,
                                'log_path': str(log_path.relative_to(ROOT)),
                                'epoch': int(m.group('epoch')),
                                'loss': float(m.group('loss')),
                                'acc@1': float(m.group('acc')),
                                'precision@1': float(m.group('precision')),
                                'recall@1': float(m.group('recall')),
                                'f1@1': float(m.group('f1')),
                            })
            except Exception as e:
                # 忽略不可读文件
                pass

# 写出 CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.DictWriter(
        fout,
        fieldnames=['dataset_or_group','model','exp','log_path','epoch','loss','acc@1','precision@1','recall@1','f1@1']
    )
    writer.writeheader()
    for r in sorted(rows, key=lambda x: (x['dataset_or_group'], x['model'], x['exp'], x['epoch'])):
        writer.writerow(r)

print(f'Written: {OUT_CSV}')
