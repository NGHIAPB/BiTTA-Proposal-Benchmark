# -*- coding: utf-8 -*-
"""Test cho utils.active_memory.ActivePriorityFIFO (bo nho S+/S-/U cua
BiTTA/BiTTA-Proposal). Instance la tuple (feat, cls, domain, entropy)."""
import conf
from utils.active_memory import ActivePriorityFIFO


def test_ba_ngan_doc_lap_nhau():
    mem = ActivePriorityFIFO(capacity=10)
    mem.add_correct_instance(("f0", 1, "d0", 0.1))
    mem.add_wrong_instance(("f1", 2, "d0", 0.2))
    mem.add_u_instance(("f2", 3, "d0", 0.3))

    assert mem.get_occupancy(mem.get_correct_memory()) == 1
    assert mem.get_occupancy(mem.get_wrong_memory()) == 1
    assert mem.get_occupancy(mem.get_u_memory()) == 1
    # them vao wrong_mem khong duoc lam thay doi correct_mem
    assert mem.get_correct_memory()[1] == [1]


def test_pop_min_loai_entropy_thap_nhat_khi_day():
    # cot entropy la cot thu 4 (chi so 3); pop="min" -> loai entropy nho nhat
    mem = ActivePriorityFIFO(capacity=2, pop="min")
    mem.add_correct_instance(("f0", 0, "d", 0.9))
    mem.add_correct_instance(("f1", 0, "d", 0.1))  # entropy nho nhat hien tai
    mem.add_correct_instance(("f2", 0, "d", 0.5))  # day -> loai f1 (entropy=0.1)

    feats = mem.get_correct_memory()[0]
    assert feats == ["f0", "f2"]


def test_pop_max_loai_entropy_cao_nhat_khi_day():
    mem = ActivePriorityFIFO(capacity=2, pop="max")
    mem.add_correct_instance(("f0", 0, "d", 0.1))
    mem.add_correct_instance(("f1", 0, "d", 0.9))  # entropy lon nhat hien tai
    mem.add_correct_instance(("f2", 0, "d", 0.5))  # day -> loai f1 (entropy=0.9)

    feats = mem.get_correct_memory()[0]
    assert feats == ["f0", "f2"]


def test_khong_delay_chen_ngay_vao_dung_ngan():
    mem = ActivePriorityFIFO(capacity=10, delay=0)
    mem.add_correct_instance(("f0", 1, "d0", 0.1))
    # delay=0 -> khong qua delayed_queue, chen truc tiep
    assert mem.delayed_queue == []
    assert mem.get_occupancy(mem.get_correct_memory()) == 1


def test_delay_tam_giu_trong_hang_doi_roi_moi_chen(monkeypatch):
    # update_every_x=1 -> moi 1 mau khong nhan la 1 "lo", kich hoat kiem tra hang doi
    monkeypatch.setattr(conf.args, "update_every_x", 1, raising=False)
    mem = ActivePriorityFIFO(capacity=10, delay=2)

    mem.add_correct_instance(("f0", 1, "d0", 0.1))
    assert mem.get_occupancy(mem.get_correct_memory()) == 0  # con dang cho trong hang doi
    assert len(mem.delayed_queue) == 1

    mem.add_u_instance(("fu0", 0, "d0", 0.0))  # lo thu 1 ke tu luc hoi -> chua du delay=2
    assert mem.get_occupancy(mem.get_correct_memory()) == 0

    mem.add_u_instance(("fu1", 0, "d0", 0.0))  # lo thu 2 -> du delay=2, duoc chen
    assert mem.get_occupancy(mem.get_correct_memory()) == 1
    assert mem.delayed_queue == []


def test_reset_xoa_ca_ba_ngan():
    mem = ActivePriorityFIFO(capacity=10)
    mem.add_correct_instance(("f0", 1, "d0", 0.1))
    mem.add_wrong_instance(("f1", 2, "d0", 0.2))
    mem.add_u_instance(("f2", 3, "d0", 0.3))

    mem.reset()

    assert mem.get_occupancy(mem.get_correct_memory()) == 0
    assert mem.get_occupancy(mem.get_wrong_memory()) == 0
    assert mem.get_occupancy(mem.get_u_memory()) == 0
