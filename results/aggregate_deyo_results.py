# -*- coding: utf-8 -*-
# Tong hop ket qua cua cac runner ghi log dang van ban qua dong "Result under ...
# average: X.XXXXX" (DeYO/TENT/EATA/SAR qua src/methods/deyo/main.py, va MEMO qua
# src/methods/memo/main.py) -- khac dinh dang online_eval.json cua ho BiTTA-family.
# Cach dung: python aggregate_deyo_results.py --log_dir ../log/deyo_deyo_ImageNet-C_normal_2024
#            python aggregate_deyo_results.py --log_dir ../log/memo_memo_cifar10_s0
import argparse
import glob
import os
import re

PAT_AVG = re.compile(r"Result under (\S+)\. The adaptation accuracy of (\S+) is\s+average:\s*([\d.]+)")
PAT_TOP1 = re.compile(r"Result under (\S+)\. The adaptation accuracy of (\S+) is top1:\s*([\d.]+)")


def parse_logfile(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for m in PAT_AVG.finditer(text):
        rows.append((m.group(1), m.group(2), float(m.group(3))))
    if not rows:
        for m in PAT_TOP1.finditer(text):
            rows.append((m.group(1), m.group(2), float(m.group(3))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True,
                     help="Thu muc --output da dung khi chay src/methods/deyo/main.py")
    args = ap.parse_args()

    txts = glob.glob(os.path.join(args.log_dir, "*.txt"))
    if not txts:
        print(f"Khong tim thay file .txt log trong {args.log_dir}")
        return

    all_rows = []
    for t in txts:
        all_rows.extend(parse_logfile(t))

    if not all_rows:
        print("Khong parse duoc dong ket qua nao -- kiem tra lai dinh dang log (DeYO co "
              "the da doi format o phien ban khac).")
        return

    print(f"{'Corruption':<25}{'Method':<10}{'Accuracy (%)':>15}")
    print("-" * 50)
    accs = []
    for corr, method, acc in all_rows:
        print(f"{corr:<25}{method:<10}{acc*100 if acc <= 1 else acc:>15.3f}")
        accs.append(acc*100 if acc <= 1 else acc)
    print("-" * 50)
    print(f"{'TRUNG BINH':<35}{sum(accs)/len(accs):>15.3f}   (n={len(accs)})")


if __name__ == "__main__":
    main()
