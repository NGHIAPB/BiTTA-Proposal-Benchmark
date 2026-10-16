# -*- coding: utf-8 -*-
"""Cau hinh chung cho toan bo test suite.

Cac module loi (utils.active_memory, learner.dnn, ...) doc conf.args ngay
luc import (vd torch.device(..., conf.args.gpu_idx)), nen conf.args phai
duoc gan TRUOC khi bat ky test module nao import chung -- vi vay viec gan
duoc thuc hien o muc module cua conftest.py (chay mot lan, truoc khi
pytest thu thap/import cac file test_*.py), khong phai trong mot fixture.

Test suite nay la unit test NHE, chay duoc tren CPU, KHONG can GPU, KHONG
can tai dataset/checkpoint that -- khac voi scripts/smoke_test.sh (chay
toan bo pipeline that tren du lieu/mo hinh that, can GPU + checkpoint).
"""
import os
import sys
import types

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import conf  # noqa: E402

if conf.args is None:
    # Namespace toi thieu du de import cac module trong src/ khong loi.
    # Chi dien cac truong ma code doc o muc MODULE-LEVEL (vd active_memory.py,
    # learner/dnn.py doc conf.args.gpu_idx khi import) -- test tung ham/lop cu
    # the se tu dien them cac truong khac neu can qua conf.args truc tiep.
    conf.args = types.SimpleNamespace(
        gpu_idx=0,
        seed=0,
        update_every_x=64,
    )
