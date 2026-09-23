# Adapter noi DeYO (runner rieng, cwd = src/methods/deyo/) voi 5 dataset da co san cua ho BiTTA-family
# (src/data_loader/*.py) de Continuous/Mixed shift dung DUNG mot bo du lieu da chuan bi tren Kaggle cho
# ca 5 phuong phap, khong phai chuan bi rieng cho DeYO. Day la lua chon co chu dich KHAC voi cach vendor
# DeYO nguyen ven (ImageNet-C/WaterBirds/ColoredMNIST da co san): phan THUAT TOAN cua DeYO (methods/deyo.py,
# tent.py, eata.py, sar.py, sam.py) giu nguyen 100% khong dung toi; chi phan DOC DU LIEU va KIEN TRUC MO
# HINH duoc noi lai voi code da kiem thu cua ho BiTTA-family (bat buoc de nap dung checkpoint chung),
# thay vi viet lai 5 lan (giam rui ro loi vat ly du lieu/kien truc).
#
# Vi sao can chdir: cac lop CIFAR10Dataset/CIFAR100Dataset/TinyImageNetDataset/PacsDataset/ImageNetRDataset
# doc duong dan tuong doi TU CHINH conf.py (vd conf.CIFAR10Opt['file_path'] = './dataset/CIFAR-10-C'),
# KHONG dung tham so `file=` truyen vao __init__ (dac diem co san cua code goc, khong phai loi cua ban
# tich hop nay). DeYO chay voi cwd = src/methods/deyo/, nen phai chdir tam ve thu muc goc project khi
# khoi tao dataset/mo hinh (cac lenh torch.load checkpoint cung dung duong dan tuong doi goc project).
import os
import random
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))          # .../TTA-Proposal-Benchmark/src
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))                   # .../TTA-Proposal-Benchmark
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@contextmanager
def _at_repo_root():
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


# corruption_list dung chung cho 3 dataset kieu "-C" (giong CORR15 trong scripts/run_bitta_family.sh)
CORR15 = ["gaussian_noise-5", "shot_noise-5", "impulse_noise-5", "defocus_blur-5", "glass_blur-5",
          "motion_blur-5", "zoom_blur-5", "snow-5", "frost-5", "fog-5", "brightness-5", "contrast-5",
          "elastic_transform-5", "pixelate-5", "jpeg_compression-5"]

# dset_key (--dset cua DeYO) -> danh sach don vi de lap qua trong Continuous/Mixed
SCENARIO_UNITS = {
    "CIFAR10-C": CORR15,
    "CIFAR100-C": CORR15,
    "TinyImageNet-C": CORR15,
    "PACS-scenario": ["art_painting", "cartoon", "sketch"],
    "ImageNetR-scenario": ["corrupt"],
    "WaterBirds-scenario": ["test"],
    "ColoredMNIST-scenario": ["test"],
}


class ScenarioValDataset:
    """Boc 1 trong 5 Dataset cua ho BiTTA-family thanh dinh dang DeYO can: __getitem__ tra ve it nhat
    (anh, nhan); co san .switch_mode()/.set_dataset_size() de tuong thich validate()/Fisher cua DeYO
    (giong quy uoc WaterbirdsDataset/ColoredMNIST da co san trong repo DeYO)."""

    def __init__(self, dset_key, unit):
        self.dset_key = dset_key
        with _at_repo_root():
            if dset_key == "CIFAR10-C":
                from data_loader.CIFAR10Dataset import CIFAR10Dataset
                self.ds = CIFAR10Dataset(domains=[unit], max_source=9999, transform="val")
            elif dset_key == "CIFAR100-C":
                from data_loader.CIFAR100Dataset import CIFAR100Dataset
                self.ds = CIFAR100Dataset(domains=[unit], max_source=9999, transform="val")
            elif dset_key == "TinyImageNet-C":
                from data_loader.TINYIMAGENETDataset import TinyImageNetDataset
                self.ds = TinyImageNetDataset(domain=unit, max_source=9999, transform="val")
            elif dset_key == "PACS-scenario":
                from data_loader.PACSDataset import PacsDataset
                self.ds = PacsDataset(domains=[unit], max_source=9999, transform="val")
            elif dset_key == "ImageNetR-scenario":
                from data_loader.IMAGENETRDataset import ImageNetRDataset
                self.ds = ImageNetRDataset(domain=unit, max_source=9999, transform="val")
            elif dset_key == "WaterBirds-scenario":
                from data_loader.WATERBIRDSDataset import WaterbirdsDataset
                self.ds = WaterbirdsDataset(file="", domains=[unit], max_source=9999, transform="val")
            elif dset_key == "ColoredMNIST-scenario":
                import torchvision.transforms as transforms
                import conf
                from data_loader.COLOREDMNISTDataset import ColoredMNISTDataset
                self.ds = ColoredMNISTDataset(root=conf.COLORED_MNIST["file_path"], env=unit,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))]))
            else:
                raise NotImplementedError(dset_key)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Tiny-ImageNet-C/PACS/ImageNet-R doc anh LAZY qua torchvision.ImageFolder, luu duong dan
        # TUONG DOI luc __init__ -> phai chdir lai moi lan __getitem__(), khong chi luc khoi tao.
        with _at_repo_root():
            img, cl, _dl = self.ds[idx]
        if self.dset_key == "WaterBirds-scenario":
            return img, cl, _dl   # phan tu thu 3 = place (nen anh), dung cho do chinh xac 4 nhom cua DeYO
        return img, cl

    def switch_mode(self, original, rotation):
        pass

    def set_dataset_size(self, n):
        """Dung boi buoc tinh Fisher cua EATA (fisher_dataset.set_dataset_size(args.fisher_size)):
        cat ngau nhien (seed co dinh) con toi da n phan tu de gioi han so anh tinh Fisher."""
        import torch.utils.data as tud
        n = min(n, len(self.ds))
        idx = list(range(len(self.ds)))
        random.Random(0).shuffle(idx)
        self.ds = tud.Subset(self.ds, idx[:n])
        return n


_CKPT_BY_KEY = {
    "CIFAR10-C": ("pretrained_weights/cifar10/cp_last_{seed}.pth.tar", 10, False),
    "CIFAR100-C": ("pretrained_weights/cifar100/cp_last_{seed}.pth.tar", 100, False),
    "TinyImageNet-C": ("pretrained_weights/tiny-imagenet/cp_last_{seed}.pth.tar", 200, True),
    "PACS-scenario": ("pretrained_weights/pacs/cp_last.pth.tar", 7, True),
    "WaterBirds-scenario": ("pretrained_weights/waterbirds/cp_last.pth.tar", 2, True),
    "ColoredMNIST-scenario": ("pretrained_weights/colored-mnist/cp_last_{seed}.pth.tar", 2, True),
}


def _load_by_path(mod_name, rel_path):
    # DeYO co package rieng ten `models`/`utils` (trong src/methods/deyo/) -> `import models.ResNet` se
    # tro nham vao do. Nap file cua BiTTA-family bang duong dan tuyet doi, dat ten module rieng.
    import importlib.util
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(SRC_DIR, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Lop chuan hoa (mean/std) ma dnn.py boc NGOAI mo hinh SAU KHI nap checkpoint (xem get_normalize_layer);
# DeYO khong chuan hoa trong transform nen phai boc o day, neu khong do chinh xac se sai.
_NORM_KEY = {"CIFAR10-C": "cifar10", "CIFAR100-C": "cifar100", "TinyImageNet-C": "tiny-imagenet",
             "PACS-scenario": "pacs", "ImageNetR-scenario": "imagenetR", "WaterBirds-scenario": "waterbirds",
             "ColoredMNIST-scenario": "colored-mnist"}


def build_scenario_model(dset_key, seed, device):
    """Dung DUNG checkpoint/kien truc/chuan hoa ma ho BiTTA-family da dung cho dataset nay (xem
    scripts/run_bitta_family.sh), de ca 7 phuong phap trong de an xuat phat tu cung 1 bo trong so."""
    with _at_repo_root():
        rn = _load_by_path("_bitta_resnet", os.path.join("models", "ResNet.py"))
        nl = _load_by_path("_bitta_normalize_layer", os.path.join("utils", "normalize_layer.py"))
        if dset_key == "ImageNetR-scenario":
            # Khong can checkpoint rieng: ResNet-18 pretrained torchvision, loc 1000->200 logit.
            import conf
            import torchvision
            net = rn.ResNetDropout18(filter=conf.IMAGENET_R["indices_in_1k"])
            pre = torchvision.models.resnet18(pretrained=True)
            net.load_state_dict(pre.state_dict())
        else:
            if dset_key not in _CKPT_BY_KEY:
                raise NotImplementedError(dset_key)
            ckpt_tpl, num_class, pretrained_arch = _CKPT_BY_KEY[dset_key]
            ckpt_path = ckpt_tpl.format(seed=seed)
            net = (rn.ResNetDropout18() if pretrained_arch else rn.ResNet18())
            net.fc = nn.Linear(512, num_class)
            state = torch.load(ckpt_path, map_location="cpu")
            net.load_state_dict(state, strict=True)
        norm = nl.get_normalize_layer(_NORM_KEY[dset_key])
        net = nn.Sequential(norm, net)
    return net.to(device)
