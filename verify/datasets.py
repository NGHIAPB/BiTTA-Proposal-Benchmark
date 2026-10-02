# -*- coding: utf-8 -*-
"""Cau hinh rieng cho tung dataset (khop CHINH XAC voi scripts/run_bitta_family.sh) + ham nap anh
SACH that (chua nhieu) cho Tang 1 + Tang 2. Nhieu Gaussian duoc ap THEM vao sau, giong nhau cho ca
4 dataset (xem common.gaussian_noise), KHONG phai file corrupted .npy/.jpg chinh thuc cua tung
benchmark -- xem README.md.
"""
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ============================================================ cau hinh rieng tung dataset
# dropout_rate / n_dropouts / arch lay DUNG tu scripts/run_bitta_family.sh (DROPOUT/NDROP/MODEL).
# norm_mean/std: cifar10/100 dung thong ke rieng (utils/normalize_layer.py); tiny-imagenet VA pacs
# deu dung thong ke ImageNet (tiny-imagenet chuan hoa trong model qua NormalizeLayer; pacs chuan hoa
# ngay trong transform cua PACSDataset.py -- ca hai duong deu cho CUNG mot phep tinh, nen o day gop
# chung lam MOT buoc duy nhat truoc khi vao mang, khong quan trong no "logic" nam o dau).
_CIFAR10_MEAN, _CIFAR10_STD = [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]
_CIFAR100_MEAN, _CIFAR100_STD = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
_IMAGENET_MEAN, _IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

CONFIGS = {
    "cifar10_c": dict(
        arch="resnet18", num_class=10, dropout_rate=0.3, n_dropouts=4, img_size=32,
        norm_mean=_CIFAR10_MEAN, norm_std=_CIFAR10_STD,
        ckpt=lambda repo, seed: os.path.join(repo, "pretrained_weights", "cifar10", f"cp_last_{seed}.pth.tar"),
        seeds=[0, 1, 2], needs_data_root=False,
    ),
    "cifar100_c": dict(
        arch="resnet18", num_class=100, dropout_rate=0.3, n_dropouts=4, img_size=32,
        norm_mean=_CIFAR100_MEAN, norm_std=_CIFAR100_STD,
        ckpt=lambda repo, seed: os.path.join(repo, "pretrained_weights", "cifar100", f"cp_last_{seed}.pth.tar"),
        seeds=[0, 1, 2], needs_data_root=False,
    ),
    "tiny_imagenet_c": dict(
        arch="resnet18_pretrained", num_class=200, dropout_rate=0.1, n_dropouts=2, img_size=224,
        norm_mean=_IMAGENET_MEAN, norm_std=_IMAGENET_STD,
        ckpt=lambda repo, seed: os.path.join(repo, "pretrained_weights", "tiny-imagenet", f"cp_last_{seed}.pth.tar"),
        seeds=[0, 1, 2], needs_data_root=True,
        data_root_hint="dataset/Tiny-ImageNet-C/origin/Data/train (thu muc 200 lop dung de huan luyen checkpoint nguon)",
    ),
    "pacs": dict(
        arch="resnet18_pretrained", num_class=7, dropout_rate=0.3, n_dropouts=2, img_size=224,
        norm_mean=_IMAGENET_MEAN, norm_std=_IMAGENET_STD,
        ckpt=lambda repo, seed: os.path.join(repo, "pretrained_weights", "pacs", "cp_last.pth.tar"),
        seeds=[0, 1, 2], needs_data_root=True,  # chi 1 checkpoint dung chung -> 3 "seed" la 3 lan rut nhieu khac nhau
        data_root_hint="domainbed_dataset/PACS/photo (mien nguon dung de huan luyen checkpoint)",
    ),
}


def load_clean_cifar(name, data_root, nsample, seed):
    cls = torchvision.datasets.CIFAR10 if name == "cifar10_c" else torchvision.datasets.CIFAR100
    ds = cls(root=data_root, train=False, download=True)
    imgs = ds.data.astype(np.float32) / 255.0  # (N,H,W,3) trong [0,1]
    labels = np.array(ds.targets, dtype=np.int64)
    if nsample and nsample < len(imgs):
        idx = np.random.RandomState(seed).choice(len(imgs), nsample, replace=False)
        imgs, labels = imgs[idx], labels[idx]
    return imgs, labels


def load_clean_imagefolder(data_root, img_size, nsample, seed):
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    full = torchvision.datasets.ImageFolder(data_root, transform=transform)
    n_total = len(full)
    if nsample and nsample < n_total:
        idx = np.random.RandomState(seed).choice(n_total, nsample, replace=False).tolist()
        ds = Subset(full, idx)
    else:
        ds = full
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    imgs_list, labels_list = [], []
    for x, y in loader:
        imgs_list.append(x.permute(0, 2, 3, 1).numpy())  # CHW -> HWC, van con trong [0,1] (chi ToTensor)
        labels_list.append(y.numpy())
    imgs = np.concatenate(imgs_list).astype(np.float32)
    labels = np.concatenate(labels_list).astype(np.int64)
    return imgs, labels


def load_clean(dataset_key, data_root, nsample, seed):
    if dataset_key in ("cifar10_c", "cifar100_c"):
        return load_clean_cifar(dataset_key, data_root, nsample, seed)
    cfg = CONFIGS[dataset_key]
    return load_clean_imagefolder(data_root, cfg["img_size"], nsample, seed)
