# -*- coding: utf-8 -*-
"""Tang 1 + Tang 2: kiem chung tien de "confidence thap -> thuong sai, cao -> thuong dung" cua
BiTTA goc, tren 1 trong 4 dataset: cifar10_c | cifar100_c | tiny_imagenet_c | pacs.

Cach dung (chay tu thu muc goc TTA-Proposal-Benchmark, hoac tu chinh thu muc verify/):
    python verify/premise_check.py --dataset cifar10_c
    python verify/premise_check.py --dataset cifar100_c
    python verify/premise_check.py --dataset tiny_imagenet_c --data-root dataset/Tiny-ImageNet-C/origin/Data/train --nsample 5000
    python verify/premise_check.py --dataset pacs --data-root domainbed_dataset/PACS/photo

Xem verify/README.md de biet chi tiet tung tham so va cach chuan bi du lieu.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

VERIFY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VERIFY_DIR)
sys.path.insert(0, VERIFY_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from datasets import CONFIGS, load_clean  # noqa: E402
from common import (gaussian_noise, run_stream, bin_and_entropy, print_table,  # noqa: E402
                    save_csv, save_plot)
from models.ResNet import ResNet18, ResNetDropout18  # noqa: E402
from utils.normalize_layer import NormalizeLayer  # noqa: E402


def build_model(cfg, ckpt_path, device):
    if cfg["arch"] == "resnet18":
        net = ResNet18()
    elif cfg["arch"] == "resnet18_pretrained":
        net = ResNetDropout18()
    else:
        raise NotImplementedError(cfg["arch"])
    net.fc = nn.Linear(net.fc.in_features, cfg["num_class"])
    state = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(state, strict=True)

    norm = NormalizeLayer(cfg["norm_mean"], cfg["norm_std"])
    model = nn.Sequential(norm, net)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = True  # tuong duong --use_learned_stats
    model.eval()
    return model.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--data-root", default=None,
                     help="Bat buoc voi tiny_imagenet_c/pacs (xem data_root_hint trong datasets.py). "
                          "Voi cifar10_c/cifar100_c: noi luu cache tai ve (mac dinh verify/.cache).")
    ap.add_argument("--sigma", type=float, default=0.26, help="Do lech chuan nhieu Gaussian tren thang [0,1].")
    ap.add_argument("--nsample", type=int, default=None,
                     help="Gioi han so anh SACH ban dau truoc khi them nhieu (mac dinh: toan bo voi "
                          "cifar, 5000 voi tiny_imagenet_c, toan bo voi pacs vi mien 'photo' von da nho).")
    ap.add_argument("--update-every-x", type=int, default=64, help="Kich thuoc batch cap nhat BN (giong --update_every_x).")
    ap.add_argument("--bn-momentum", type=float, default=0.3, help="Giong --bn_momentum cua BiTTA/BiTTA-Proposal.")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--sanity-check-n", type=int, default=200,
                     help="So anh SACH dung de kiem tra nhanh truoc moi seed (phat hien lech thu tu lop).")
    ap.add_argument("--skip-sanity-check", action="store_true", help="Bo qua buoc kiem tra anh sach.")
    ap.add_argument("--out-dir", default=os.path.join(VERIFY_DIR, "results"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = CONFIGS[args.dataset]
    if cfg["needs_data_root"] and not args.data_root:
        raise SystemExit(f"--data-root la bat buoc voi dataset '{args.dataset}'. "
                          f"Goi y: {cfg.get('data_root_hint', '?')}")
    data_root = args.data_root or os.path.join(VERIFY_DIR, ".cache")
    os.makedirs(args.out_dir, exist_ok=True)

    nsample_default = {"tiny_imagenet_c": 5000}.get(args.dataset, None)
    nsample = args.nsample if args.nsample is not None else nsample_default

    print(f"Dataset: {args.dataset} | device: {args.device} | sigma nhieu: {args.sigma}")
    print(f"Nap anh sach (co dinh, khong doi theo seed) ...")
    clean_imgs, labels = load_clean(args.dataset, data_root, nsample, seed=0)
    print(f"So anh sach: {len(clean_imgs)}")
    labels_t = torch.from_numpy(labels)

    all_C, all_correct = [], []
    for seed in cfg["seeds"]:
        ckpt_path = cfg["ckpt"](REPO_ROOT, seed)
        if not os.path.exists(ckpt_path):
            print(f"[BO QUA] khong thay checkpoint: {ckpt_path}")
            continue
        rng = np.random.RandomState(seed)
        noisy = gaussian_noise(clean_imgs, args.sigma, rng)
        imgs_t = torch.from_numpy(noisy).permute(0, 3, 1, 2).contiguous()

        model = build_model(cfg, ckpt_path, args.device)

        if not args.skip_sanity_check:
            # Kiem tra nhanh tren ANH SACH (chua them nhieu), KHONG cap nhat BN, dung dung running
            # stats trong checkpoint -- neu thu tu lop (class order) giua checkpoint va thu muc anh
            # lech nhau, accuracy se ~ngau nhien (1/num_class), bao dong truoc khi phan tich sai ca loat.
            n_check = min(args.sanity_check_n, len(clean_imgs))
            idx = np.random.RandomState(0).choice(len(clean_imgs), n_check, replace=False)
            chk_imgs = torch.from_numpy(clean_imgs[idx]).permute(0, 3, 1, 2).contiguous().to(args.device)
            chk_labels = labels_t[idx].to(args.device)
            with torch.no_grad():
                chk_pred = model(chk_imgs).argmax(dim=1)
            chk_acc = (chk_pred == chk_labels).float().mean().item()
            chance = 1.0 / cfg["num_class"]
            flag = "  <-- CANH BAO: gan muc ngau nhien, co the LECH THU TU LOP!" if chk_acc < 3 * chance else ""
            print(f"    [kiem tra anh sach] accuracy tren {n_check} anh KHONG nhieu = {chk_acc:.3f} "
                  f"(muc ngau nhien = {chance:.3f}){flag}")
            if chk_acc < 3 * chance:
                print("    -> Dung lai, kiem tra lai --data-root (thu tu lop co the khong khop checkpoint). "
                      "Dung --skip-sanity-check neu muon bo qua canh bao nay.")
                continue

        print(f"--- seed {seed} (checkpoint: {os.path.basename(ckpt_path)}) ---")
        C, correct = run_stream(model, imgs_t, labels_t, args.update_every_x, cfg["n_dropouts"],
                                cfg["dropout_rate"], cfg["num_class"], args.bn_momentum, args.device,
                                log_prefix="    ")
        all_C.append(C)
        all_correct.append(correct)

    if not all_C:
        raise SystemExit("Khong co seed nao chay duoc (kiem tra lai checkpoint).")

    C_all = np.concatenate(all_C)
    correct_all = np.concatenate(all_correct)
    print(f"\nTong so cap (C, dung/sai) gop tu {len(all_C)} seed: {len(C_all)}")

    rows = bin_and_entropy(C_all, correct_all, n_bins=args.n_bins)
    print_table(rows)

    tag = f"{args.dataset}_gaussian_sigma{args.sigma:.2f}".replace(".", "")
    save_csv(rows, os.path.join(args.out_dir, f"premise_check_{tag}.csv"))
    save_plot(rows, f"{args.dataset} — Gaussian σ={args.sigma}",
             os.path.join(args.out_dir, f"premise_check_{tag}.png"))


if __name__ == "__main__":
    main()
