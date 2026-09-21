# -*- coding: utf-8 -*-
# Tong hop ket qua TENT*/EATA*/SAR*/BiTTA/BiTTA-Proposal thanh bang so sanh theo (dataset, setting).
# (* = phien ban co binary feedback, xem scripts/run_bitta_family.sh)
#
# Doc log tu   log/<ds>/<TENT|EATA|SAR|BiTTA>/tgt_<...>/<log_prefix>/.../online_eval.json  voi
#   log_prefix = <method>_<dataset>_<setting>_s<seed>   (mac dinh cua run_bitta_family.sh)
#   continuous : tgt_cont/<prefix>/<unit>/online_eval.json   -> trung binh accuracy cuoi cua tung unit
#   mixed      : tgt_cont/<prefix>/random/online_eval.json   -> accuracy tich luy tai 25/50/75/100% luong du lieu
#   fully      : tgt_<unit>/<prefix>/online_eval.json        -> trung binh accuracy cuoi cua tung unit
# unit = corruption (CIFAR-C, Tiny-ImageNet-C), domain (PACS), corrupt (ImageNet-R), test (ColoredMNIST).
#
# Cach dung: python aggregate_results.py --dataset cifar10_c --setting continuous
#            python aggregate_results.py --all
import argparse
import json
import math
import os
import numpy as np

CORR15 = ["gaussian_noise-5", "shot_noise-5", "impulse_noise-5", "defocus_blur-5",
          "glass_blur-5", "motion_blur-5", "zoom_blur-5", "snow-5", "frost-5",
          "fog-5", "brightness-5", "contrast-5", "elastic_transform-5",
          "pixelate-5", "jpeg_compression-5"]
PACS3 = ["art_painting", "cartoon", "sketch"]

# ds = ten thu muc log (= gia tri --dataset cua src/main.py)
DATASET_CFG = {
    "cifar10_c":       dict(ds="cifar10",        units=CORR15, seeds=[0, 1, 2], settings=["continuous", "mixed", "fully"]),
    "cifar100_c":      dict(ds="cifar100",       units=CORR15, seeds=[0, 1, 2], settings=["continuous", "fully"]),
    "tiny_imagenet_c": dict(ds="tiny-imagenet",  units=CORR15, seeds=[0, 1, 2], settings=["continuous", "fully"]),
    "pacs":            dict(ds="pacs",           units=PACS3,  seeds=[0, 1, 2, 3, 4], settings=["continuous", "mixed", "fully"]),
    "imagenet_r":      dict(ds="imagenetR",      units=["corrupt"], seeds=[0, 1, 2], settings=["fully"]),
    "colored_mnist":   dict(ds="colored-mnist",  units=["test"],    seeds=[0, 1, 2], settings=["fully"]),
}

METHODS = ["tent", "eata", "sar", "bitta_baseline", "bitta_proposal"]
METHOD_LABEL = {
    "tent": "TENT*", "eata": "EATA*", "sar": "SAR*",
    "bitta_baseline": "BiTTA", "bitta_proposal": "BiTTA-Proposal (C1+C4)",
}
# thu muc log = gia tri --method truyen cho src/main.py
METHOD_DIR = {"tent": "TENT", "eata": "EATA", "sar": "SAR", "bitta_baseline": "BiTTA", "bitta_proposal": "BiTTA"}
MIXED_POINTS = (0.25, 0.50, 0.75, 1.00)


def _load_acc(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    return j.get("accuracy") or None


def load_run(log_root, cfg, method, dataset, setting, seed):
    """continuous/fully: tra ve trung binh accuracy cuoi cua cac unit (None neu thieu unit).
    mixed: tra ve list accuracy tich luy tai 25/50/75/100%."""
    ds, prefix = cfg["ds"], f"{method}_{dataset}_{setting}_s{seed}"
    mdir = os.path.join(log_root, ds, METHOD_DIR[method])
    if setting == "mixed":
        acc = _load_acc(os.path.join(mdir, "tgt_cont", prefix, "random", "online_eval.json"))
        if not acc:
            return None
        return [acc[max(0, math.ceil(p * len(acc)) - 1)] for p in MIXED_POINTS]
    finals = []
    for u in cfg["units"]:
        if setting == "continuous":
            acc = _load_acc(os.path.join(mdir, "tgt_cont", prefix, u, "online_eval.json"))
        else:
            acc = _load_acc(os.path.join(mdir, f"tgt_{u}", prefix, "online_eval.json"))
        if not acc:
            return None
        finals.append(acc[-1])
    return float(np.mean(finals))


def report(log_root, dataset, setting):
    cfg = DATASET_CFG[dataset]
    seeds = cfg["seeds"]
    print(f"\n=== {dataset} | {setting} ===")
    mixed = setting == "mixed"
    head = f"{'Method':<26}" + ("".join(f"{int(p*100):>7}%" for p in MIXED_POINTS) if mixed else f"{'Mean Acc (%)':>14}")
    print(head + f"{'Std':>9}{'# seed':>10}")
    print("-" * len(head + f"{'Std':>9}{'# seed':>10}"))
    rows = []
    for m in METHODS:
        vals = [load_run(log_root, cfg, m, dataset, setting, s) for s in seeds]
        vals = [v for v in vals if v is not None]
        if not vals:
            print(f"{METHOD_LABEL[m]:<26}{'chua co du lieu':>14}")
            continue
        arr = np.array(vals, dtype=float)
        if mixed:   # arr: (seed, 4); std tinh tren accuracy cuoi (100%)
            mean, std = arr.mean(0), (arr[:, -1].std(ddof=1) if len(arr) > 1 else 0.0)
            line = f"{METHOD_LABEL[m]:<26}" + "".join(f"{x:>8.2f}" for x in mean)
            score = mean[-1]
        else:
            score = arr.mean(); std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            line = f"{METHOD_LABEL[m]:<26}{score:>14.3f}"
        print(line + f"{std:>9.3f}{len(arr):>7}/{len(seeds)}")
        rows.append((METHOD_LABEL[m], score))
    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"Tot nhat: {best[0]} ({best[1]:.3f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_CFG.keys()))
    ap.add_argument("--setting", choices=["continuous", "mixed", "fully"])
    ap.add_argument("--all", action="store_true", help="in moi to hop (dataset, setting) cua kich ban")
    ap.add_argument("--log_root", default="../log")
    args = ap.parse_args()

    if args.all:
        for d, cfg in DATASET_CFG.items():
            for s in cfg["settings"]:
                report(args.log_root, d, s)
        return
    if not args.dataset or not args.setting:
        ap.error("can --dataset va --setting (hoac --all)")
    if args.setting not in DATASET_CFG[args.dataset]["settings"]:
        ap.error(f"to hop khong thuoc kich ban: {args.dataset} khong chay setting {args.setting}")
    report(args.log_root, args.dataset, args.setting)


if __name__ == "__main__":
    main()
