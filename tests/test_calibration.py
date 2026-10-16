# -*- coding: utf-8 -*-
"""Test cho utils.calibration.expected_calibration_error.

Luu y: ham nay doi dau vao la torch.Tensor (dung .abs() cua Tensor, khong
phai cua numpy.ndarray) -- da kiem chung truc tiep truoc khi viet test nay."""
import pytest
import torch

from utils.calibration import expected_calibration_error


def test_tra_ve_none_khi_rong():
    conf_list = torch.tensor([])
    correct_list = torch.tensor([])
    assert expected_calibration_error(conf_list, correct_list, num_bins=10, order=1) is None


def test_ece_tinh_tay_voi_2_bin_doi_xung():
    # 4 mau: 2 mau conf=0.1 deu SAI, 2 mau conf=0.9 deu DUNG.
    # Tinh tay (xem docstring module): ca hai gia tri tra ve deu = 0.1.
    conf_list = torch.tensor([0.1, 0.1, 0.9, 0.9])
    correct_list = torch.tensor([0.0, 0.0, 1.0, 1.0])

    ece, label_ece = expected_calibration_error(conf_list, correct_list, num_bins=10, order=1)

    assert ece == pytest.approx(0.1, abs=1e-6)
    assert label_ece == pytest.approx(0.1, abs=1e-6)


def test_hieu_chuan_hoan_hao_cho_ece_bang_0():
    # conf du doan dung bang chinh ty le dung thuc te trong tung bin -> ECE = 0.
    conf_list = torch.tensor([0.5, 0.5])
    correct_list = torch.tensor([1.0, 0.0])  # accuracy trong bin = 0.5 = confidence

    ece, _ = expected_calibration_error(conf_list, correct_list, num_bins=10, order=1)
    assert ece == pytest.approx(0.0, abs=1e-6)
