#!/usr/bin/env python
import os
import re
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = ROOT / 'experiments' / 'validation'

# 仅匹配 log-testN.txt，N 为 0..19（忽略大小写）；排除包含 plain 的文件
FNAME_RE = re.compile(r'^log-test(?P<idx>([0-9]|1[0-9]))\.txt$', re.IGNORECASE)
EXCLUDE_KEYWORD = 'plain'

# 匹配指标行：包含 loss/acc/precision/recall/f1
METRIC_RE = re.compile(
    r"loss\s+(?P<loss>[0-9.]+)\s+"
    r"acc@1\s+(?P<acc>[0-9.]+)\s+"
    r"precision@1\s+(?P<precision>[0-9.]+)\s+"
    r"recall@1\s+(?P<recall>[0-9.]+)\s+"
    r"f1@1\s+(?P<f1>[0-9.]+)",
    re.IGNORECASE
)

# 导出列，顺序固定
OUT_FIELDS = ['exp','test','acc@1','precision@1','recall@1','f1@1','loss']

def parse_metrics_from_file(log_path: Path):
    last = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = METRIC_RE.search(line)
                if m:
                    last = {
                        'loss': float(m.group('loss')),
                        'acc@1': float(m.group('acc')),
                        'precision@1': float(m.group('precision')),
                        'recall@1': float(m.group('recall')),
                        'f1@1': float(m.group('f1')),
                    }
    except Exception:
        return None
    return last

def main():
    if not VAL_DIR.exists():
        print('validation dir not found:', VAL_DIR)
        return

    groups = {}

    for root, dirs, files in os.walk(VAL_DIR):
        p = Path(root)
        for name in files:
            lname = name.lower()
            if EXCLUDE_KEYWORD in lname:
                continue
            m = FNAME_RE.match(lname)
            if not m:
                continue
            test_idx = m.group('idx')

            log_path = p / name
            try:
                rel = log_path.relative_to(VAL_DIR).parts
                # 期望结构: <dataset>/<model>/<exp>/log-testN.txt
                dataset = rel[0] if len(rel) > 0 else ''
                model = rel[1] if len(rel) > 1 else ''
                exp = rel[2] if len(rel) > 2 else ''
            except Exception:
                dataset = ''
                model = ''
                exp = ''

            metrics = parse_metrics_from_file(log_path)
            if not metrics:
                continue

            key = (dataset, model)
            row = {
                'exp': exp,
                'test': test_idx,
                'acc@1': f"{metrics['acc@1']:.4f}",
                'precision@1': f"{metrics['precision@1']:.4f}",
                'recall@1': f"{metrics['recall@1']:.4f}",
                'f1@1': f"{metrics['f1@1']:.4f}",
                'loss': f"{metrics['loss']:.4f}",
            }
            groups.setdefault(key, []).append(row)

    written = []
    for (dataset, model), rows in groups.items():
        # 排序：先按 exp，再按 test 数字
        def sort_key(r):
            try:
                t = int(r.get('test','0'))
            except Exception:
                t = 0
            return (r.get('exp',''), t)
        rows_sorted = sorted(rows, key=sort_key)

        out_name = f"metrics_{dataset}_{model}_validation.csv" if dataset else f"metrics_{model}_validation.csv"
        out_path = VAL_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=OUT_FIELDS)
            writer.writeheader()
            for r in rows_sorted:
                writer.writerow(r)
        written.append(out_path)

    print('Written:')
    for p in written:
        print(' ', p)

if __name__ == '__main__':
    main()
