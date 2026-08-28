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
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torchvision.transforms as transforms

from utils.augmix import make_augmix
from utils.data import prepare_test_data

cudnn.benchmark = True


def build_model(num_class, checkpoint_path, device):
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


def adapt_single(net, image, optimizer, aug_fn, batch_size, niter, device):
    net.eval()
    for _ in range(niter):
        inputs = torch.stack([aug_fn(image) for _ in range(batch_size)]).to(device)
        optimizer.zero_grad()
        loss, _ = marginal_entropy(net(inputs))
        loss.backward()
        optimizer.step()


def test_single(net, image, label, te_transform, device):
    net.eval()
    inputs = te_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = net(inputs).argmax(dim=1).item()
    return int(predicted == label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="memo", choices=["no_adapt", "memo"])
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tiny-imagenet", "pacs"])
    parser.add_argument("--corruption", default="gaussian_noise",
                         help="ten corruption (cifar/tiny-imagenet) hoac ten domain (pacs)")
    parser.add_argument("--level", default=5, type=int, help="muc do severity 1-5 (bo qua voi pacs)")
    parser.add_argument("--data_root", default="../../../dataset")
    parser.add_argument("--domainbed_root", default="../../../domainbed_dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int,
                         help="so luong augmix augmentation moi anh test (MEMO Algorithm 1)")
    parser.add_argument("--niter", default=1, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--nsample", default=-1, type=int, help="gioi han so anh test, -1 = toan bo (dung cho smoke test)")
    parser.add_argument("--output", default="../../../log/memo")
    parser.add_argument("--gpu_idx", default=0, type=int)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    teset, meta = prepare_test_data(args.dataset, args.corruption, args.level, args.data_root, args.domainbed_root)
    aug_fn = make_augmix(meta["img_size"], meta["mean"], meta["std"])
    te_transform = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])

    net = build_model(meta["num_class"], args.checkpoint, device)
    ckpt_state = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    n = len(teset) if args.nsample < 0 else min(args.nsample, len(teset))
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, f"{args.dataset}_{args.corruption}_L{args.level}_s{args.seed}.txt")

    correct = []
    with open(log_path, "w", encoding="utf-8") as logf:
        for i in range(n):
            image, label = teset[i]
            if args.method == "memo":
                net.load_state_dict(ckpt_state)
                adapt_single(net, image, optimizer, aug_fn, args.batch_size, args.niter, device)
            correct.append(test_single(net, image, label, te_transform, device))

            if (i + 1) % 200 == 0 or i == n - 1:
                acc = 100.0 * sum(correct) / len(correct)
                msg = f"[{i + 1}/{n}] running acc: {acc:.3f}"
                print(msg)
                logf.write(msg + "\n")

        acc = sum(correct) / len(correct)
        result_line = (f"Result under {args.corruption}-{args.level}. "
                        f"The adaptation accuracy of {args.method.upper()} is  average: {acc:.5f}")
        print(result_line)
        logf.write(result_line + "\n")


if __name__ == "__main__":
    main()
