# Doc du lieu test cho MEMO tu dung cac thu muc dataset da co san cua project
# (dataset/CIFAR-10-C, dataset/CIFAR-100-C, dataset/Tiny-ImageNet-C, domainbed_dataset/PACS),
# theo dung quy uoc thu muc ma src/process_cifar.py va src/data_loader/*.py da dung --
# de checkpoint pretrain o pretrained_weights/ dung chung duoc cho ca MEMO.
#
# Chuan hoa (mean/std) LAY DUNG theo src/utils/normalize_layer.py: checkpoint cua ho
# BiTTA-family duoc train voi normalize nay (bocked trong model, khong o transform), nen
# MEMO phai ap dung chinh xac cung he so thi checkpoint moi cho ket qua dung.
import os

import numpy as np
import torch.utils.data as data
import torchvision.datasets as tv_datasets
from PIL import Image


DATASET_META = {
    "cifar10": dict(num_class=10, img_size=32,
                     mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    "cifar100": dict(num_class=100, img_size=32,
                      mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    "tiny-imagenet": dict(num_class=200, img_size=224,
                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "pacs": dict(num_class=7, img_size=224,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
}


class _NpyDataset(data.Dataset):
    """CIFAR-10-C/100-C: dataset/<name>/corrupted/severity-<level>/<corruption>.npy + labels.npy."""

    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path).astype("int64")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return Image.fromarray(self.data[idx]), int(self.labels[idx])


def prepare_test_data(dataset, corruption, level, data_root, domainbed_root):
    """Tra ve (teset, meta) voi teset[i] -> (PIL.Image, label)."""
    meta = DATASET_META[dataset]

    if dataset in ("cifar10", "cifar100"):
        root = os.path.join(data_root, "CIFAR-10-C" if dataset == "cifar10" else "CIFAR-100-C")
        if corruption == "original":
            teset = _NpyDataset(os.path.join(root, "origin", "original.npy"),
                                 os.path.join(root, "origin", "labels.npy"))
        else:
            sev_dir = os.path.join(root, "corrupted", f"severity-{level}")
            teset = _NpyDataset(os.path.join(sev_dir, f"{corruption}.npy"),
                                 os.path.join(sev_dir, "labels.npy"))
    elif dataset == "tiny-imagenet":
        root = os.path.join(data_root, "Tiny-ImageNet-C")
        path = (os.path.join(root, "origin", "Data", "val") if corruption == "original"
                else os.path.join(root, "corrupted", corruption, str(level)))
        teset = tv_datasets.ImageFolder(path, transform=None)
    elif dataset == "pacs":
        # doi voi PACS, tham so --corruption chinh la ten domain (art_painting|cartoon|sketch|photo)
        root = os.path.join(domainbed_root, "PACS")
        teset = tv_datasets.ImageFolder(os.path.join(root, corruption), transform=None)
    else:
        raise ValueError(f"Dataset khong ho tro cho MEMO: {dataset}")

    return teset, meta
