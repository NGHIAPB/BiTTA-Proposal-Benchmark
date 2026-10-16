# -*- coding: utf-8 -*-
"""Test "import khong loi" cho cac module loi trong src/learner va src/utils.

Day la test re nhung co gia tri: cac module nay thuc hien cong viec o muc
MODULE-LEVEL luc import (vd doc conf.args.gpu_idx, goi torch.cuda.*), nen
mot thay doi vo y o do co the lam HONG VIEC IMPORT tren may khong co GPU
ma khong co bat ky dong code nao khac bao loi -- dung nhu loi
torch.cuda.set_device() khong dieu kien da phat hien va sua trong
learner/dnn.py va utils/active_memory.py (xem lich su sua doi)."""
import importlib

import pytest

MODULES = [
    "utils.memory",
    "utils.active_memory",
    "utils.calibration",
    "utils.normalize_layer",
    "utils.loss_functions",
    "utils.reset_utils",
    "models.ResNet",
    "learner.dnn",
    "learner.tent",
    "learner.sar",
    "learner.eata",
    "learner.bitta",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_import_khong_loi(module_name):
    importlib.import_module(module_name)
