#!/usr/bin/env python
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / 'experiments' / 'train'

OUT_FIELDS = ['epoch','acc@1','f1@1','recall@1','precision@1','loss']

def moving_average(values, window=5):
    smoothed = []
    if not values:
        return smoothed
    for i in range(len(values)):
        start = max(0, i - window + 1)
        sub = values[start:i+1]
        smoothed.append(sum(sub) / len(sub))
    return smoothed

def process_file(csv_path: Path, window=5):
    try:
        rows = []
        with open(csv_path, 'r', encoding='utf-8', newline='') as fin:
            reader = csv.DictReader(fin)
            for r in reader:
                rows.append(r)
        # 获取 loss 并平滑
        losses = []
        for r in rows:
            try:
                losses.append(float(r.get('loss','')))
            except Exception:
                losses.append(None)
        # 简单前向填充 None
        last = 0.0
        for i, v in enumerate(losses):
            if v is None:
                losses[i] = last
            last = losses[i]
        smoothed = moving_average(losses, window=window)
        # 覆盖写回
        for r, s in zip(rows, smoothed):
            r['loss'] = f"{s:.4f}"
        with open(csv_path, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=OUT_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return True
    except Exception:
        return False


def main():
    if not TRAIN_DIR.exists():
        print('train dir not found:', TRAIN_DIR)
        return
    matched = list(TRAIN_DIR.glob('metrics_*.csv'))
    if not matched:
        print('no per-model metrics CSV found in:', TRAIN_DIR)
        return
    ok = 0
    for p in matched:
        if process_file(p, window=5):
            ok += 1
            print('smoothed:', p)
        else:
            print('failed:', p)
    print(f'done. smoothed {ok}/{len(matched)} files')

if __name__ == '__main__':
    main()
