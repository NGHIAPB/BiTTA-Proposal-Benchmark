# -*- coding: utf-8 -*-
"""Test cho utils.memory.FIFO (bo nho mot ngan dung boi TENT/EATA/SAR)."""
from utils.memory import FIFO


def test_add_instance_tang_occupancy():
    mem = FIFO(capacity=3)
    assert mem.get_occupancy() == 0
    mem.add_instance(("feat0", "cls0", "dom0"))
    assert mem.get_occupancy() == 1
    assert mem.get_memory() == [["feat0"], ["cls0"], ["dom0"]]


def test_vuot_capacity_loai_phan_tu_cu_nhat_theo_fifo():
    mem = FIFO(capacity=2)
    mem.add_instance(("f0", "c0", "d0"))
    mem.add_instance(("f1", "c1", "d1"))
    mem.add_instance(("f2", "c2", "d2"))  # vuot capacity -> loai f0/c0/d0

    assert mem.get_occupancy() == 2
    assert mem.get_memory() == [["f1", "f2"], ["c1", "c2"], ["d1", "d2"]]


def test_remove_instance_by_index_giua():
    mem = FIFO(capacity=5)
    for i in range(4):
        mem.add_instance((f"f{i}", f"c{i}", f"d{i}"))

    mem.remove_instance_by_index(1)  # loai phan tu o vi tri 1 (f1/c1/d1)

    assert mem.get_memory() == [["f0", "f2", "f3"], ["c0", "c2", "c3"], ["d0", "d2", "d3"]]


def test_reset_tra_ve_rong():
    mem = FIFO(capacity=3)
    mem.add_instance(("f0", "c0", "d0"))
    mem.reset()
    assert mem.get_occupancy() == 0
    assert mem.get_memory() == [[], [], []]


def test_save_va_set_state_dict_khoi_phuc_dung_du_lieu():
    mem = FIFO(capacity=2)
    mem.add_instance(("f0", "c0", "d0"))
    state = mem.save_state_dict()

    mem2 = FIFO(capacity=99)
    mem2.set_memory(state)

    assert mem2.capacity == 2
    assert mem2.get_memory() == [["f0"], ["c0"], ["d0"]]
    # save_state_dict() phai sao chep doc lap (khong chia se list voi ban goc)
    mem.add_instance(("f1", "c1", "d1"))
    assert mem2.get_occupancy() == 1
