# -*- coding: utf-8 -*-
"""Ham dung chung cho Tang 1 + Tang 2 (kiem chung tien de "confidence thap -> thuong sai, cao ->
thuong dung" cua BiTTA goc), ap dung giong nhau cho ca 4 dataset trong datasets.py.

Confidence duoc tinh SAU buoc cap nhat thong ke BatchNorm theo tung batch (update_every_x anh),
dung cau hinh that cua BiTTA/BiTTA-Proposal (--use_learned_stats --bn_momentum 0.3) -- giong dung
trang thai ma active_sample_selection() trong learner/dnn.py thuc su nhin thay. Khong mo phong
buoc cap nhat trong so bang gradient (BFA/ABA) tu vai mau phan hoi moi batch -- day la hieu ung
bac hai so voi cap nhat BatchNorm, va co y bo qua de khong gan ket qua vao 1 thuat toan chon mau
cu the nao (xem README.md trong thu muc nay).
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_bn_track_running_stats(model):
    """Tuong duong --use_learned_stats: BN dung trung binh cong don (EMA) thay vi chi dung batch hien tai."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = True


def set_bn_momentum(model, momentum):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum


def update_bn_stats(model, images, bn_momentum):
    """Tuong duong pre_active_sample_selection(): cho batch qua mang o che do train de BN cap nhat
    running_mean/var voi momentum=bn_momentum, roi dong lai (momentum=0) truoc khi suy luan."""
    set_bn_momentum(model, bn_momentum)
    model.train()
    with torch.no_grad():
        _ = model(images)
    set_bn_momentum(model, 0.0)
    model.eval()


def confidence_and_correctness(model, images, labels, n_dropouts, dropout_rate, num_class, device):
    """Tra ve (C(x), dung/sai) cho ca batch, dung DUNG cong thuc BiTTA: C(x) = trung binh softmax
    qua n_dropouts luot (dropout luon bat) tai lop du doan khong-dropout y*."""
    with torch.no_grad():
        y_pred = model(images).argmax(dim=1)

        probs_sum = torch.zeros(images.size(0), num_class, device=device)
        for _ in range(n_dropouts):
            feat = model[0](images)
            out = model[1](feat, dropout=dropout_rate)
            probs_sum += F.softmax(out, dim=1)
        probs_mean = probs_sum / n_dropouts
        C = probs_mean.gather(1, y_pred.view(-1, 1)).squeeze(1)
        correct = (y_pred == labels).float()
    return C.cpu().numpy(), correct.cpu().numpy()


def run_stream(model, imgs_chw, labels, update_every_x, n_dropouts, dropout_rate, num_class,
               bn_momentum, device, log_prefix=""):
    """Chay toan bo anh (da noise) theo tung batch update_every_x, cap nhat BN tuan tu (giong 1 luong
    continual TTA that), tra ve mang C(x) va dung/sai gop lai tu toan bo luong."""
    Cs, corrects, running_acc = [], [], []
    n = imgs_chw.size(0)
    for i in range(0, n, update_every_x):
        batch_imgs = imgs_chw[i:i + update_every_x].to(device)
        batch_labels = labels[i:i + update_every_x].to(device)
        update_bn_stats(model, batch_imgs, bn_momentum)
        C, correct = confidence_and_correctness(model, batch_imgs, batch_labels, n_dropouts,
                                                 dropout_rate, num_class, device)
        Cs.append(C)
        corrects.append(correct)
        running_acc.append(correct.mean())
    if running_acc:
        print(f"{log_prefix}accuracy 5 batch dau = {[round(float(a), 3) for a in running_acc[:5]]}, "
              f"5 batch cuoi = {[round(float(a), 3) for a in running_acc[-5:]]}, "
              f"trung binh ca luong = {np.mean(running_acc):.3f}")
    return np.concatenate(Cs), np.concatenate(corrects)


def bin_and_entropy(C_all, correct_all, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (C_all >= lo) & (C_all < hi) if b < n_bins - 1 else (C_all >= lo) & (C_all <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, np.nan, np.nan, np.nan))
            continue
        acc = float(correct_all[mask].mean())
        mean_c = float(C_all[mask].mean())
        p = min(max(acc, 1e-6), 1 - 1e-6)
        hb = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
        rows.append((lo, hi, n, mean_c, acc, hb))
    return rows


def print_table(rows):
    print("\n" + "=" * 78)
    print(f"{'Khoang C(x)':<14}{'So mau':>8}{'C trung binh':>14}{'Accuracy thuc nghiem':>22}{'h_b (nat)':>12}")
    print("-" * 78)
    for lo, hi, n, mean_c, acc, hb in rows:
        rng_str = f"[{lo:.1f},{hi:.1f}{']' if hi == 1.0 else ')'}"
        if n == 0:
            print(f"{rng_str:<14}{n:>8}{'--':>14}{'--':>22}{'--':>12}")
        else:
            print(f"{rng_str:<14}{n:>8}{mean_c:>14.3f}{acc:>22.3f}{hb:>12.4f}")
    print("=" * 78)

    valid = [r for r in rows if r[2] > 0]
    if valid:
        peak = max(valid, key=lambda r: r[5])
        lowest = valid[0]
        print(f"\nBin thap nhat (noi BiTTA goc luon chon):  C ~ [{lowest[0]:.1f},{lowest[1]:.1f}) "
              f"-> accuracy = {lowest[4]:.3f}, h_b = {lowest[5]:.4f}")
        print(f"Bin co h_b CAO NHAT (nhieu thong tin nhat): C ~ [{peak[0]:.1f},{peak[1]:.1f}) "
              f"-> accuracy = {peak[4]:.3f}, h_b = {peak[5]:.4f}")


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lo", "hi", "n", "mean_confidence", "empirical_accuracy", "h_b"])
        w.writerows(rows)
    print(f"Da luu bang chi tiet: {path}")


def save_plot(rows, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Khong ve duoc hinh (bo qua):", e)
        return
    xs = [(r[0] + r[1]) / 2 for r in rows]
    accs = [r[4] for r in rows]
    hbs = [r[5] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(title)
    ax1.plot(xs, accs, "o-", color="#1f3a5f")
    ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="hiệu chuẩn hoàn hảo")
    ax1.set_xlabel("Confidence MC-Dropout C(x)")
    ax1.set_ylabel("Accuracy thực nghiệm")
    ax1.set_title("Tầng 1: đường cong hiệu chuẩn")
    ax1.legend(fontsize=8)
    ax2.plot(xs, hbs, "o-", color="#c00000")
    ax2.axvline(xs[0], color="gray", linestyle=":", label="bin BiTTA gốc luôn chọn")
    ax2.set_xlabel("Confidence MC-Dropout C(x)")
    ax2.set_ylabel("Entropy nhị phân h_b (nat)")
    ax2.set_title("Tầng 2: thông tin kỳ vọng mỗi câu hỏi")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Da luu hinh: {path}")


def gaussian_noise(imgs_float01, sigma, rng):
    """imgs_float01: numpy (N,H,W,3) float32 trong [0,1]."""
    x = imgs_float01 + rng.normal(0, sigma, size=imgs_float01.shape).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)
