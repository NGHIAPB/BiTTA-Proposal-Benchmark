# Doc du lieu test cho MEMO tu dung cac thu muc dataset da co san cua project
# (dataset/CIFAR-10-C, dataset/CIFAR-100-C, dataset/Tiny-ImageNet-C, domainbed_dataset/PACS,
# dataset/imagenet-r, dataset/ColoredMNIST, dataset/WaterBirds), theo dung quy uoc thu muc ma
# src/process_cifar.py va src/data_loader/*.py da dung -- de checkpoint pretrain o
# pretrained_weights/ dung chung duoc cho ca MEMO.
#
# Chuan hoa (mean/std) LAY DUNG theo src/utils/normalize_layer.py: checkpoint cua ho
# BiTTA-family duoc train voi normalize nay (bocked trong model, khong o transform), nen
# MEMO phai ap dung chinh xac cung he so thi checkpoint moi cho ket qua dung.
import csv
import glob
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
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
    # ImageNet-R: khong can checkpoint (--checkpoint imagenet_r), ResNet-18 pretrained torchvision
    # loc 1000->200 logit -- xem _FilteredLogits trong main.py.
    "imagenet_r": dict(num_class=200, img_size=224,
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ColoredMNIST: 2 lop, anh 28x28, checkpoint tu huan luyen (--method Src trong ho BiTTA-family).
    "colored_mnist": dict(num_class=2, img_size=28,
                           mean=[0.1307, 0.1307, 0.], std=[0.3081, 0.3081, 0.3081]),
    # WaterBirds: 2 lop, bao cao them do chinh xac theo 4 nhom (groups=True), xem main.py.
    "waterbirds": dict(num_class=2, img_size=224,
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], groups=True),
}


class _ImageNetRDataset(data.Dataset):
    """ImageNet-R cho MEMO: ImageFolder tren dataset/imagenet-r (200 thu muc wnid, nhan = thu tu sap xep)."""

    def __init__(self, root):
        self.base = tv_datasets.ImageFolder(root, transform=None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


class _ColoredMNISTDataset(data.Dataset):
    """ColoredMNIST cho MEMO: chi doc split 'test' (dataset/ColoredMNIST/test.pt), tu chua (khong import
    src/data_loader/COLOREDMNISTDataset.py). Dinh dang tung phan tu: (PIL anh, nhan nhieu, mau_do, nhan_goc);
    voi split test, nhan_nhieu == nhan_goc (xem COLOREDMNISTDataset.prepare_colored_mnist) nen lay item[1]."""

    def __init__(self, root, split="test"):
        if split != "test":
            raise ValueError("MEMO chi ho tro ColoredMNIST split 'test' (--corruption test)")
        path = os.path.join(root, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Khong thay {path} -- chay truoc 1 lan qua ho BiTTA-family (vd Fully TTA) de "
                "COLOREDMNISTDataset.py tu tao train1.pt/train2.pt/test.pt tu MNIST.")
        self.data = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx][0], self.data[idx][1]
        return img, int(label)

WB_SPLIT_ID = {"train": 0, "val": 1, "test": 2}
WB_H5_GROUP = "Waterbirds"


class _NpyDataset(data.Dataset):
    """CIFAR-10-C/100-C: dataset/<name>/corrupted/severity-<level>/<corruption>.npy + labels.npy."""

    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path).astype("int64")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return Image.fromarray(self.data[idx]), int(self.labels[idx])


class _WaterbirdsDataset(data.Dataset):
    """WaterBirds cho MEMO: tra ve (PIL 224x224, y); thuoc tinh .places[i] la nen anh (0 land / 1 water).

    Dung LAI dung 2 dinh dang ma ho BiTTA-family/DeYO doc (xem src/data_loader/WATERBIRDSDataset.py,
    tu chua o day de MEMO khong import cheo): file *.h5py (do pretrain_waterbirds.py tao) hoac thu muc
    co metadata.csv + anh. group = 2 * y + place -> LL/LS/SL/SS.
    """

    def __init__(self, root, split):
        if split not in WB_SPLIT_ID:
            raise ValueError("Split WaterBirds khong hop le: %r (chi co %s) -- truyen --corruption test"
                             % (split, list(WB_SPLIT_ID)))
        self.split = split
        self._h5 = None
        parent = os.path.dirname(os.path.normpath(root))
        bases = [root] + [os.path.join(parent, n) for n in ("WaterBirds", "Waterbirds", "waterbirds")]
        self.kind = self.path = None
        for base in bases:
            if not os.path.isdir(base):
                continue
            dirs = [base] + sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
            for d in dirs:
                hits = [h for pat in ("*.h5py", "*.h5", "*.hdf5") for h in sorted(glob.glob(os.path.join(d, pat)))]
                if hits:
                    self.kind, self.path = "h5", hits[0]
                    break
            if self.kind is None:
                for d in dirs:
                    if os.path.isfile(os.path.join(d, "metadata.csv")):
                        self.kind, self.path = "csv", os.path.join(d, "metadata.csv")
                        break
            if self.kind is not None:
                break
        if self.kind is None:
            raise FileNotFoundError("Khong tim thay du lieu WaterBirds trong %s (can *.h5py hoac metadata.csv)" % root)

        if self.kind == "h5":
            import h5py
            with h5py.File(self.path, "r") as f:
                grp = f[WB_H5_GROUP][split]
                n = len(grp)
                self.labels = np.array([int(grp[str(i)].attrs["y"]) for i in range(n)], dtype=np.int64)
                self.places = np.array([int(grp[str(i)].attrs["place"]) for i in range(n)], dtype=np.int64)
        else:
            base_dir, want = os.path.dirname(self.path), WB_SPLIT_ID[split]
            files, labels, places = [], [], []
            with open(self.path, "r", newline="") as fh:
                rdr = csv.reader(fh)
                next(rdr)
                for row in rdr:
                    if int(row[3]) == want:
                        files.append(os.path.join(base_dir, row[1]))
                        labels.append(int(row[2]))
                        places.append(int(row[4]))
            self._files = files
            self.labels = np.array(labels, dtype=np.int64)
            self.places = np.array(places, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.kind == "h5":
            if self._h5 is None:
                import h5py
                self._h5 = h5py.File(self.path, "r")
            img = Image.fromarray(np.asarray(self._h5[WB_H5_GROUP][self.split][str(idx)][()], dtype=np.uint8))
        else:
            img = Image.open(self._files[idx]).convert("RGB")
        if img.size != (224, 224):
            img = img.resize((224, 224))
        return img, int(self.labels[idx])


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
    elif dataset == "imagenet_r":
        # 1 domain dich duy nhat ("corrupt"), bo qua gia tri --corruption/--level thuc te
        teset = _ImageNetRDataset(os.path.join(data_root, "imagenet-r"))
    elif dataset == "colored_mnist":
        # 1 domain dich duy nhat ("test"), bo qua gia tri --level
        teset = _ColoredMNISTDataset(os.path.join(data_root, "ColoredMNIST"), split=corruption)
    elif dataset == "waterbirds":
        # doi voi WaterBirds, --corruption la ten split (mac dinh dung "test")
        teset = _WaterbirdsDataset(os.path.join(data_root, "WaterBirds"), corruption)
    elif dataset == "pacs":
        # doi voi PACS, tham so --corruption chinh la ten domain (art_painting|cartoon|sketch|photo)
        root = os.path.join(domainbed_root, "PACS")
        teset = tv_datasets.ImageFolder(os.path.join(root, corruption), transform=None)
    else:
        raise ValueError(f"Dataset khong ho tro cho MEMO: {dataset}")

    return teset, meta
