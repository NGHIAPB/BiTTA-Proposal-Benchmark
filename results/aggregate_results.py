# -*- coding: utf-8 -*-
# Tong hop ket qua TENT/EATA/SAR/BiTTA-baseline/BiTTA-Proposal thanh bang so sanh
# kieu Table 1 cua paper BiTTA/DeYO. Doc log tu
# log/<dataset>/BiTTA/tgt_cont/<log_prefix>/<unit>/online_eval.json
# Cach dung: python aggregate_results.py --dataset cifar10_c
import argparse
import json
import os
import numpy as np

CORR15 = ["gaussian_noise-5", "shot_noise-5", "impulse_noise-5", "defocus_blur-5",
          "glass_blur-5", "motion_blur-5", "zoom_blur-5", "snow-5", "frost-5",
          "fog-5", "brightness-5", "contrast-5", "elastic_transform-5",
          "pixelate-5", "jpeg_compression-5"]

DATASET_CFG = {
    "cifar10_c":       dict(ds="cifar10",       units=CORR15, seeds=[0, 1, 2]),
    "cifar100_c":      dict(ds="cifar100",      units=CORR15, seeds=[0, 1, 2]),
    "tiny_imagenet_c":  dict(ds="tiny-imagenet", units=CORR15, seeds=[0, 1, 2]),
    "pacs":            dict(ds="pacs", units=["art_painting", "cartoon", "sketch"],
                             seeds=[0, 1, 2, 3, 4]),
}

METHODS = ["tent", "eata", "sar", "bitta_baseline", "bitta_proposal"]
METHOD_LABEL = {
    "tent": "TENT", "eata": "EATA", "sar": "SAR",
    "bitta_baseline": "BiTTA (baseline)", "bitta_proposal": "BiTTA-Proposal (C1+C4)",
}


def load_run(log_root, ds, log_prefix, units):
    acc = {}
    base = os.path.join(log_root, ds, "BiTTA", "tgt_cont", log_prefix)
    for u in units:
        f = os.path.join(base, u, "online_eval.json")
        if os.path.exists(f):
            j = json.load(open(f))
            if j.get("accuracy"):
                acc[u] = j["accuracy"][-1]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CFG.keys()))
    ap.add_argument("--log_root", default="../log")
    args = ap.parse_args()

    cfg = DATASET_CFG[args.dataset]
    ds, units, seeds = cfg["ds"], cfg["units"], cfg["seeds"]

    print(f"{'Method':<28}{'Mean Acc (%)':>15}{'Std':>10}{'# seed hoan tat':>18}")
    print("-" * 71)
    rows = []
    for m in METHODS:
        seed_means = []
        for s in seeds:
            log_prefix = f"{m}_{args.dataset}_s{s}"
            acc = load_run(args.log_root, ds, log_prefix, units)
            if len(acc) == len(units):
                seed_means.append(np.mean(list(acc.values())))
        if seed_means:
            arr = np.array(seed_means)
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            print(f"{METHOD_LABEL[m]:<28}{arr.mean():>15.3f}{std:>10.3f}{len(arr):>15}/{len(seeds)}")
            rows.append((METHOD_LABEL[m], arr.mean(), std, len(arr)))
        else:
            print(f"{METHOD_LABEL[m]:<28}{'chua co du lieu':>15}")

    if rows:
        best = max(rows, key=lambda r: r[1])
        print("-" * 71)
        print(f"Tot nhat: {best[0]} ({best[1]:.3f}%)")


if __name__ == "__main__":
    main()
