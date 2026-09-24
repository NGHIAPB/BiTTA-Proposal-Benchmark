# MEMO (Zhang, Levine, Finn -- NeurIPS 2022, "Test Time Robustness via Adaptation
# and Augmentation", arXiv:2110.09506). Entrypoint RIENG cho MEMO (giong DeYO), vi
# thuat toan la episodic: voi MOI anh test, nap lai checkpoint goc, augment thanh
# 1 batch augmix, toi thieu hoa marginal entropy tren batch do, roi du doan --
# khac han vong lap adapt() lien tuc (continual) cua ho BiTTA-family/DeYO.
#
# Dung LAI checkpoint ResNet-18 cua ho BiTTA-family (pretrained_weights/<dataset>/
# cp_last_<seed>.pth.tar) de dam bao ca 7 phuong phap trong de an xuat phat tu cung
# 1 bo trong so pretrained -- xem build_model() ben duoi, sao chep dung trinh tu
# thay fc + chuan hoa cua src/learner/dnn.py (KHONG import cheo sang ho BiTTA-family).
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torchvision.transforms as transforms

from utils.augmix import make_augmix
from utils.data import prepare_test_data

cudnn.benchmark = True

# ResNet-18 KIEU CIFAR (conv1 3x3, khong maxpool, avg_pool2d(4)) -- sao chep tu src/models/ResNet.py
# (BasicBlock + ResNet) de MEMO tu chua, khong import cheo sang ho BiTTA-family. Checkpoint
# pretrained_weights/cifar10|cifar100/cp_last_<seed>.pth.tar duoc train tren kien truc NAY, KHONG PHAI
# torchvision.models.resnet18 (conv1 7x7 + maxpool) -- nap nham kien truc se loi lech shape state_dict.
class _CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class _CifarResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(_CifarBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return self.fc(out.view(out.size(0), -1))


# 200 chi so lop ImageNet-1000 tuong ung 200 lop ImageNet-R -- sao chep tu conf.IMAGENET_R['indices_in_1k']
# (src/conf.py) de MEMO tu chua, khong import cheo sang conf cua ho BiTTA-family.
_IMAGENET_R_INDICES = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99,
    100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187,
    195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276,
    277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337,
    338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425,
    428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579,
    583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787,
    805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945,
    947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]


class _FilteredLogits(nn.Module):
    """Boc 1 mang 1000 lop, chi giu lai cac logit trong `indices` (dung cho ImageNet-R: 1000->200)."""

    def __init__(self, base, indices):
        super().__init__()
        self.base = base
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, x):
        return self.base(x)[:, self.indices]


def build_model(dataset, num_class, checkpoint_path, device):
    if checkpoint_path == "imagenet":
        # Khong co checkpoint rieng, dung ResNet-18 pretrained cua torchvision (giong
        # `--model resnet18_pretrained` cua ho BiTTA-family, khong nap checkpoint nao them).
        assert num_class == 1000, "--checkpoint imagenet chi dung khi dataset co 1000 lop"
        try:
            net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:  # torchvision < 0.13
            net = tv_models.resnet18(pretrained=True)
        return net.to(device)
    if checkpoint_path == "imagenet_r":
        # Khong can checkpoint: ResNet-18 pretrained torchvision (1000 lop), loc con 200 logit
        # tuong ung 200 lop ImageNet-R -- giong het cach ResNetDropout(filter=...) cua ho BiTTA-family.
        assert num_class == 200, "--checkpoint imagenet_r chi dung khi dataset co 200 lop (imagenet_r)"
        try:
            base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        except AttributeError:
            base = tv_models.resnet18(pretrained=True)
        return _FilteredLogits(base, _IMAGENET_R_INDICES).to(device)
    if dataset in ("cifar10", "cifar100"):
        # SUA: checkpoint cifar10/cifar100 cua ho BiTTA-family la ResNet-18 KIEU CIFAR (conv1 3x3),
        # khong phai torchvision.models.resnet18 (conv1 7x7) -- truoc day nap nham kien truc se loi.
        net = _CifarResNet18(num_classes=num_class)
    else:
        net = tv_models.resnet18(num_classes=1000)
        num_feats = net.fc.in_features
        net.fc = nn.Linear(num_feats, num_class)
    ckpt = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(ckpt, strict=True)
    return net.to(device)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def adapt_single(net, image, optimizer, aug_fn, batch_size, niter, device, tm=None):
    net.eval()
    for _ in range(niter):
        t0 = time.perf_counter()
        inputs = torch.stack([aug_fn(image) for _ in range(batch_size)])
        t1 = time.perf_counter()
        inputs = inputs.to(device)
        optimizer.zero_grad()
        loss, _ = marginal_entropy(net(inputs))
        loss.backward()
        optimizer.step()
        if tm is not None:   # chi khi --profile: tach thoi gian augmix (CPU) va forward/backward (GPU)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tm["augmix_cpu"] += t1 - t0
            tm["fwd_bwd_gpu"] += time.perf_counter() - t1


def test_single(net, image, label, te_transform, device):
    net.eval()
    inputs = te_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = net(inputs).argmax(dim=1).item()
    return int(predicted == label)


def finalize(args, meta, correct, groups, logf):
    """Ghi dong ket qua cuoi (dung chung cho chay 1 tien trinh va gop shard)."""
    acc = sum(correct) / len(correct)
    tag = f"{args.corruption}" if meta.get("groups") else f"{args.corruption}-{args.level}"
    if meta.get("groups"):
        # WaterBirds: bao cao giong DeYO -- trung binh 4 nhom LL/LS/SL/SS (bo qua nhom khong co anh)
        grp = np.array(groups)
        ok = np.array(correct, dtype=float)
        gacc = [ok[grp == g].mean() if (grp == g).any() else float("nan") for g in range(4)]
        valid = [a for a in gacc if not np.isnan(a)]
        detail = ", ".join(f"{nm}: {a:.5f}" for nm, a in zip(("LL", "LS", "SL", "SS"), gacc))
        detail_line = f"- Detailed result under {tag}. {detail}, raw_acc: {acc:.5f}, worst: {min(valid):.5f}"
        print(detail_line)
        logf.write(detail_line + "\n")
        acc = float(np.mean(valid))
    result_line = (f"Result under {tag}. "
                    f"The adaptation accuracy of {args.method.upper()} is  average: {acc:.5f}")
    print(result_line)
    logf.write(result_line + "\n")


def _prefix(args):
    return os.path.join(args.output, f"{args.dataset}_{args.corruption}_L{args.level}_s{args.seed}")


def merge_shards(args):
    """Gop K shard (moi anh doc lap nen gop dung tung anh) thanh file ket qua cuoi <prefix>.txt."""
    from utils.data import DATASET_META
    meta = DATASET_META[args.dataset]
    K = args.merge_shards
    correct, groups = [], []
    for i in range(K):
        with open(f"{_prefix(args)}.shard{i}of{K}.json", encoding="utf-8") as fh:
            d = json.load(fh)
        correct += d["correct"]
        groups += d["groups"]
    with open(_prefix(args) + ".txt", "w", encoding="utf-8") as logf:
        msg = f"[gop {K} shard, {len(correct)} anh]"
        print(msg)
        logf.write(msg + "\n")
        finalize(args, meta, correct, groups, logf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="memo", choices=["no_adapt", "memo"])
    parser.add_argument("--dataset", required=True,
                         choices=["cifar10", "cifar100", "tiny-imagenet", "pacs", "imagenet_r", "colored_mnist", "waterbirds"])
    parser.add_argument("--corruption", default="gaussian_noise",
                         help="ten corruption (cifar/tiny-imagenet), ten domain (pacs), hoac ten split co dinh "
                              "('corrupt' cho imagenet_r, 'test' cho colored_mnist/waterbirds -- chi 1 lua chon)")
    parser.add_argument("--level", default=5, type=int, help="muc do severity 1-5 (bo qua voi pacs)")
    parser.add_argument("--data_root", default="../../../dataset")
    parser.add_argument("--domainbed_root", default="../../../domainbed_dataset")
    parser.add_argument("--checkpoint", required=True,
                         help="duong dan .pth.tar (state_dict ResNet-18) HOAC 'imagenet' = torchvision pretrained")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int,
                         help="so luong augmix augmentation moi anh test (MEMO Algorithm 1)")
    parser.add_argument("--niter", default=1, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--nsample", default=-1, type=int,
                         help="gioi han so anh test, -1 = toan bo. Khi gioi han, lay NGAU NHIEN (theo --seed) "
                              "thay vi n anh dau (ImageFolder sap theo lop nen n anh dau chi thuoc vai lop)")
    parser.add_argument("--output", default="../../../log/memo")
    parser.add_argument("--gpu_idx", default=0, type=int)
    parser.add_argument("--num_shards", default=1, type=int,
                         help="chia tap anh thanh K phan chay song song (moi anh doc lap -> ket qua tuong duong)")
    parser.add_argument("--shard_id", default=0, type=int, help="chi so phan [0, num_shards)")
    parser.add_argument("--merge_shards", default=0, type=int,
                         help="K > 0: chi gop K shard da chay xong thanh file ket qua cuoi, khong chay mo hinh")
    parser.add_argument("--profile", default=0, type=int,
                         help="N > 0: do thoi gian N anh dau (augmix CPU / fwd-bwd GPU / test) roi thoat, khong ghi ket qua")
    args = parser.parse_args()
    print(args)

    if args.merge_shards > 0:
        merge_shards(args)
        return
    assert 0 <= args.shard_id < args.num_shards

    seed = args.seed + args.shard_id   # shard khac nhau dung luong ngau nhien augmix khac nhau
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    teset, meta = prepare_test_data(args.dataset, args.corruption, args.level, args.data_root, args.domainbed_root)
    aug_fn = make_augmix(meta["img_size"], meta["mean"], meta["std"])
    te_transform = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])

    net = build_model(args.dataset, meta["num_class"], args.checkpoint, device)
    ckpt_state = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    n = len(teset) if args.nsample < 0 else min(args.nsample, len(teset))
    if n < len(teset):
        order = np.sort(np.random.RandomState(args.seed).choice(len(teset), n, replace=False))
    else:
        order = np.arange(len(teset))
    if args.num_shards > 1:
        order = order[args.shard_id::args.num_shards]   # cac shard roi nhau, hop lai du tap anh
        n = len(order)
    os.makedirs(args.output, exist_ok=True)
    sharded = args.num_shards > 1
    # shard ghi .log/.json (KHONG .txt) de bo tong hop khong nham voi ket qua cuoi
    log_path = (f"{_prefix(args)}.shard{args.shard_id}of{args.num_shards}.log" if sharded else _prefix(args) + ".txt")

    # nap lai checkpoint bang chep tai cho vao cac tensor cua mo hinh (ket qua giong load_state_dict, nhanh hon)
    live = list(net.state_dict().values())
    ref = list(ckpt_state.values())

    def restore():
        with torch.no_grad():
            for a, b in zip(live, ref):
                a.copy_(b)

    tm = {"augmix_cpu": 0.0, "fwd_bwd_gpu": 0.0, "restore": 0.0, "test": 0.0} if args.profile else None
    correct = []
    groups = []  # chi WaterBirds: nhom 2*y + place cua tung anh da danh gia
    with open(log_path, "w", encoding="utf-8") as logf:
        for step, i in enumerate(order):
            if tm is not None and step >= args.profile:
                per = {k: 1000.0 * v / args.profile for k, v in tm.items()}
                print("[profile] ms/anh trung binh tren %d anh: %s | tong %.1f ms/anh -> uoc tinh %.1f phut cho %d anh"
                      % (args.profile, {k: round(v, 1) for k, v in per.items()}, sum(per.values()),
                         sum(per.values()) * n / 60000.0, n))
                return
            image, label = teset[int(i)]
            if args.method == "memo":
                t0 = time.perf_counter()
                restore()
                if tm is not None:
                    tm["restore"] += time.perf_counter() - t0
                adapt_single(net, image, optimizer, aug_fn, args.batch_size, args.niter, device, tm)
            t0 = time.perf_counter()
            correct.append(test_single(net, image, label, te_transform, device))
            if tm is not None:
                tm["test"] += time.perf_counter() - t0
            if meta.get("groups"):
                groups.append(2 * int(label) + int(teset.places[int(i)]))

            if (step + 1) % 200 == 0 or step == n - 1:
                acc = 100.0 * sum(correct) / len(correct)
                msg = f"[{step + 1}/{n}] running acc: {acc:.3f}"
                print(msg)
                logf.write(msg + "\n")

        if sharded:
            with open(f"{_prefix(args)}.shard{args.shard_id}of{args.num_shards}.json", "w", encoding="utf-8") as fh:
                json.dump({"correct": correct, "groups": groups}, fh)
            msg = f"[shard {args.shard_id}/{args.num_shards}] xong {len(correct)} anh"
            print(msg)
            logf.write(msg + "\n")
        else:
            finalize(args, meta, correct, groups, logf)


if __name__ == "__main__":
    main()
