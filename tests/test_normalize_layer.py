# -*- coding: utf-8 -*-
"""Test cho utils.normalize_layer (NormalizeLayer dung trong moi learner
de chuan hoa anh dau vao, xem Chuong 2 bao cao: CIFAR dung thong ke rieng,
cac bo con lai dung thong ke ImageNet)."""
import torch

from utils.normalize_layer import (
    NormalizeLayer, IdentityLayer, get_normalize_layer,
    _CIFAR10_MEAN, _CIFAR10_STDDEV, _IMAGENET_MEAN,
)


def test_normalize_layer_cong_thuc_dung():
    means = [0.5, 0.5, 0.5]
    sds = [0.5, 0.5, 0.5]
    layer = NormalizeLayer(means, sds)

    x = torch.ones(1, 3, 2, 2)  # anh toan gia tri 1.0
    out = layer(x)
    expected = (1.0 - 0.5) / 0.5  # = 1.0

    assert torch.allclose(out, torch.full_like(out, expected))


def test_identity_layer_giu_nguyen_dau_vao():
    layer = IdentityLayer()
    x = torch.randn(2, 3, 4, 4)
    assert torch.equal(layer(x), x)


def test_get_normalize_layer_chon_dung_bo_thong_ke_theo_dataset():
    cifar10_layer = get_normalize_layer("cifar10")
    assert isinstance(cifar10_layer, NormalizeLayer)
    assert torch.allclose(cifar10_layer.mu.flatten(), torch.tensor(_CIFAR10_MEAN))

    tiny_layer = get_normalize_layer("tiny-imagenet")
    assert isinstance(tiny_layer, NormalizeLayer)
    assert torch.allclose(tiny_layer.mu.flatten(), torch.tensor(_IMAGENET_MEAN))

    # bo du lieu khong nam trong danh sach (vd pacs) -> IdentityLayer
    # (PACS tu chuan hoa trong transform cua data loader, khong o trong model)
    assert isinstance(get_normalize_layer("pacs"), IdentityLayer)
