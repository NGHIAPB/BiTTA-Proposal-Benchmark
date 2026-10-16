# -*- coding: utf-8 -*-
"""Test shape dau ra cua hai kien truc dung trong du an (xem Bang 2.1 bao
cao): ResNet18 (CIFAR-style, anh 32x32, huan luyen tu dau) va
ResNetDropout18 (chuan torchvision, anh 224x224, khoi tao tu ImageNet,
co MC-Dropout qua tham so `dropout` trong forward())."""
import torch

from models.ResNet import ResNet18, ResNetDropout18


def test_resnet18_cifar_style_dau_ra_dung_so_lop():
    net = ResNet18()
    net.fc = torch.nn.Linear(net.fc.in_features, 10)  # CIFAR-10: 10 lop
    net.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = net(x)

    assert out.shape == (2, 10)


def test_resnetdropout18_dau_ra_dung_so_lop():
    net = ResNetDropout18()
    net.fc = torch.nn.Linear(net.fc.in_features, 7)  # vd PACS: 7 lop
    net.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(x)

    assert out.shape == (2, 7)


def test_resnetdropout18_hai_lan_mc_dropout_cho_ket_qua_khac_nhau():
    # MC-Dropout (Muc 2.2.2 bao cao): dropout=0 va dropout>0 phai cho ket
    # qua khac nhau tren CUNG mot dau vao, vi forward() luon dropout o che
    # do training=True (bat ke net.eval()) khi dropout > 0.
    net = ResNetDropout18()
    net.fc = torch.nn.Linear(net.fc.in_features, 10)
    net.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out_no_dropout = net(x, dropout=0.0)
        out_with_dropout_1 = net(x, dropout=0.5)
        out_with_dropout_2 = net(x, dropout=0.5)

    assert not torch.allclose(out_no_dropout, out_with_dropout_1)
    # hai lan lay mau MC-Dropout tren cung dau vao cung phai khac nhau
    # (dung de uoc luong do khong chac chan)
    assert not torch.allclose(out_with_dropout_1, out_with_dropout_2)
