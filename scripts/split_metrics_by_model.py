#!/usr/bin/env python
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / 'experiments' / 'train'
IN_CSV = TRAIN_DIR / 'metrics_summary.csv'

# 仅保留并按此顺序输出（loss 将写入平滑后的数值）
OUT_FIELDS = ['epoch','acc@1','f1@1','recall@1','precision@1','loss']

# 输入 CSV 的字段名称
IN_FIELDS = ['dataset_or_group','model','exp','log_path','epoch','loss','acc@1','precision@1','recall@1','f1@1']

def sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)

def moving_average(values, window=5):
    smoothed = []
    if not values:
        return smoothed
    for i in range(len(values)):
        start = max(0, i - window + 1)
        sub = values[start:i+1]
        smoothed.append(sum(sub) / len(sub))
    return smoothed

def main():
    if not IN_CSV.exists():
        print(f'Input CSV not found: {IN_CSV}')
        return

    groups = {}
    with open(IN_CSV, 'r', encoding='utf-8', newline='') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            dataset = row.get('dataset_or_group','')
            model = row.get('model','')
            key = (dataset, model)
            groups.setdefault(key, []).append(row)

    written = []
    for (dataset, model), rows in groups.items():
        # 先按 exp、epoch 排序
        rows_sorted = sorted(rows, key=lambda x: (x.get('exp',''), int(x.get('epoch','0'))))

        # 按 exp 分组，对 loss 做滑动平均
        by_exp = {}
        for r in rows_sorted:
            by_exp.setdefault(r.get('exp',''), []).append(r)

        for exp, rlist in by_exp.items():
            # 收集原始 loss
            losses = []
            for r in rlist:
                try:
                    losses.append(float(r.get('loss','')))
                except Exception:
                    losses.append(None)
            # 用可用值进行平滑
            # 将 None 替换为临近已知值（简单向前填充）
            last = None
            for idx, v in enumerate(losses):
                if v is None:
                    losses[idx] = last if last is not None else 0.0
                last = losses[idx]
            smoothed = moving_average(losses, window=5)
            # 回填到 rlist 的 loss 字段
            for r, sv in zip(rlist, smoothed):
                r['loss'] = f"{sv:.4f}"

        out_name = f"metrics_{sanitize(dataset)}_{sanitize(model)}.csv"
        out_path = TRAIN_DIR / out_name
        with open(out_path, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=OUT_FIELDS)
            writer.writeheader()
            for r in rows_sorted:
                out_row = {k: r.get(k, '') for k in OUT_FIELDS}
                writer.writerow(out_row)
        written.append(out_path)

    print('Written:')
    for p in written:
        print(f'  {p}')

if __name__ == '__main__':
    main()
