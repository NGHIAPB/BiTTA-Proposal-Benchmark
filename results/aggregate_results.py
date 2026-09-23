# -*- coding: utf-8 -*-
# Tong hop ket qua TENT*/EATA*/SAR*/BiTTA/BiTTA-Proposal/DeYO/MEMO thanh bang so sanh theo (dataset, setting).
# (* = phien ban co binary feedback, xem scripts/run_bitta_family.sh)
#
# Doc log tu   log/<ds>/<TENT|EATA|SAR|BiTTA>/tgt_<...>/<log_prefix>/.../online_eval.json  voi
#   log_prefix = <method>_<dataset>_<setting>_s<seed>   (mac dinh cua run_bitta_family.sh)
#   continuous : tgt_cont/<prefix>/<unit>/online_eval.json   -> trung binh accuracy cuoi cua tung unit
#   mixed      : tgt_cont/<prefix>/random/online_eval.json   -> accuracy tich luy tai 25/50/75/100% luong du lieu
#   fully      : tgt_<unit>/<prefix>/online_eval.json        -> trung binh accuracy cuoi cua tung unit
# unit = corruption (CIFAR-C, Tiny-ImageNet-C), domain (PACS), corrupt (ImageNet-R), test (ColoredMNIST, WaterBirds).
#
# DeYO/MEMO co log dang van ban (khong co online_eval.json), doc theo dong "Result under <unit>. ... top1: X | average: X":
#   DeYO : log/deyo_deyo_<dataset>_<setting>_s<seed>/*.txt      (theo %; mixed chi co 1 so cuoi -> cot 100%)
#   MEMO : log/memo_memo_<ds>_s<seed>/<ds>_<unit>_L5_s<seed>.txt  (ti le 0-1 -> nhan 100; chi Fully)
# WaterBirds: chi so chinh = trung binh 4 nhom (avg_group); bang thu hai = nhom te nhat (worst_group), ca 7 phuong phap.
#
# Cach dung: python aggregate_results.py --dataset cifar10_c --setting continuous
#            python aggregate_results.py --all
import argparse
import json
import math
import os
import re
import glob
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
    "waterbirds":      dict(ds="waterbirds",     units=["test"],    seeds=[0, 1, 2], settings=["fully"]),
}
# ten dataset trong thu muc log cua MEMO
MEMO_DS = {"cifar10_c": "cifar10", "cifar100_c": "cifar100", "tiny_imagenet_c": "tiny-imagenet", "pacs": "pacs",
           "imagenet_r": "imagenet_r", "colored_mnist": "colored_mnist", "waterbirds": "waterbirds"}

METHODS = ["tent", "eata", "sar", "bitta_baseline", "bitta_proposal", "deyo", "memo"]
METHOD_LABEL = {
    "tent": "TENT*", "eata": "EATA*", "sar": "SAR*",
    "bitta_baseline": "BiTTA", "bitta_proposal": "BiTTA-Proposal (C1+C4)", "deyo": "DeYO", "memo": "MEMO",
}
# thu muc log = gia tri --method truyen cho src/main.py
METHOD_DIR = {"tent": "TENT", "eata": "EATA", "sar": "SAR", "bitta_baseline": "BiTTA", "bitta_proposal": "BiTTA"}
MIXED_POINTS = (0.25, 0.50, 0.75, 1.00)


def _load_acc(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    return j.get("accuracy") or None


_RES = re.compile(r"Result under (\S+?)\. The adaptation accuracy of \S+ is\s+(?:top1:\s*([\d.]+)|average:\s*([\d.]+))")
_DET = re.compile(r"Detailed result under (\S+?)\. LL: ([\d.]+), LS: ([\d.]+), SL: ([\d.]+), SS: ([\d.]+)")
_WORST = re.compile(r"Detailed result under (\S+?)\..*worst:\s*([\d.]+)")


def _parse_text_logs(pattern, scale):
    """Doc log van ban cua DeYO/MEMO -> {unit: [acc, worst]}; luot ghi sau de len luot truoc.
    worst chi co voi WaterBirds (DeYO: min 4 nhom tu dong 'Detailed'; MEMO: truong 'worst:')."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = _RES.search(line)
            if m:
                out.setdefault(m.group(1), [None, None])[0] = float(m.group(2) or m.group(3)) * scale
                continue
            m = _WORST.search(line)
            if m:
                out.setdefault(m.group(1), [None, None])[1] = float(m.group(2)) * scale
                continue
            m = _DET.search(line)
            if m:
                out.setdefault(m.group(1), [None, None])[1] = min(float(x) for x in m.groups()[1:]) * scale
    return out


def _lookup(d, u):
    for k in (u, u + "-5", u[:-2] if u.endswith("-5") else None):
        if k and k in d:
            return d[k]
    return None


def _load_text_run(log_root, cfg, method, dataset, setting, seed, metric):
    """DeYO / MEMO (log van ban). Tra ve giong load_run."""
    idx = 1 if metric == "worst" else 0
    if method == "deyo":
        d = _parse_text_logs(os.path.join(log_root, f"deyo_deyo_{dataset}_{setting}_s{seed}", "*.txt"), 1.0)
        if setting == "mixed":
            r = d.get("mix_shifts")
            return None if r is None or r[idx] is None else [float("nan")] * 3 + [r[idx]]
    else:   # memo: chi fully
        if setting != "fully":
            return None
        d = _parse_text_logs(os.path.join(log_root, f"memo_memo_{MEMO_DS[dataset]}_s{seed}", "*.txt"), 100.0)
    vals = []
    for u in cfg["units"]:
        r = _lookup(d, u)
        if r is None or r[idx] is None:
            return None
        vals.append(r[idx])
    return float(np.mean(vals))


def _valid_for(method, dataset, setting):
    if method == "memo":
        return setting == "fully"
    if method == "deyo" and setting == "mixed":
        return dataset in ("cifar10_c", "pacs")
    return True


def load_run(log_root, cfg, method, dataset, setting, seed, metric="acc"):
    """continuous/fully: tra ve trung binh accuracy cuoi cua cac unit (None neu thieu unit).
    mixed: tra ve list accuracy tich luy tai 25/50/75/100%. metric="worst": nhom te nhat (chi WaterBirds)."""
    if method in ("deyo", "memo"):
        return _load_text_run(log_root, cfg, method, dataset, setting, seed, metric)
    ds, prefix = cfg["ds"], f"{method}_{dataset}_{setting}_s{seed}"
    mdir = os.path.join(log_root, ds, METHOD_DIR[method])
    if dataset == "waterbirds":   # 4 nhom, xem src/data_loader/WATERBIRDSDataset.py
        p = os.path.join(mdir, "tgt_test", prefix, "waterbirds_groups.json")
        if not os.path.exists(p):
            return None
        return json.load(open(p))["worst_group" if metric == "worst" else "avg_group"]
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


def report(log_root, dataset, setting, metric="acc"):
    cfg = DATASET_CFG[dataset]
    seeds = cfg["seeds"]
    tag = " | nhom te nhat (worst_group)" if metric == "worst" else (" | trung binh 4 nhom" if dataset == "waterbirds" else "")
    print(f"\n=== {dataset} | {setting}{tag} ===")
    mixed = setting == "mixed"
    head = f"{'Method':<26}" + ("".join(f"{int(p*100):>7}%" for p in MIXED_POINTS) if mixed else f"{'Mean Acc (%)':>14}")
    print(head + f"{'Std':>9}{'# seed':>10}")
    print("-" * len(head + f"{'Std':>9}{'# seed':>10}"))
    rows = []
    for m in METHODS:
        if not _valid_for(m, dataset, setting):
            continue
        vals = [load_run(log_root, cfg, m, dataset, setting, s, metric) for s in seeds]
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
                if d == "waterbirds":
                    report(args.log_root, d, s, "worst")
        return
    if not args.dataset or not args.setting:
        ap.error("can --dataset va --setting (hoac --all)")
    if args.setting not in DATASET_CFG[args.dataset]["settings"]:
        ap.error(f"to hop khong thuoc kich ban: {args.dataset} khong chay setting {args.setting}")
    report(args.log_root, args.dataset, args.setting)
    if args.dataset == "waterbirds":
        report(args.log_root, args.dataset, args.setting, "worst")


if __name__ == "__main__":
    main()
