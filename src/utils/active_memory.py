import random

import numpy as np
import torch

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

class ActivePriorityFIFO:
    """Bo nho ba ngan dung cho BiTTA/BiTTA-Proposal: correct_mem (S+), wrong_mem
    (S-) va u_mem (U), tuong ung voi ba nhom mau mo ta trong luong hoat dong
    cua thuat toan (Muc 2.2.6 bao cao): mau da duoc oracle xac nhan dung,
    mau da duoc xac nhan sai, va mau chua duoc hoi (dung cho nhanh ABA).

    Moi ngan la mot danh sach 4 "cot" song song [feat, cls, domain, entropy]
    (khong phai 4 mau rieng) -- mot instance them vao la mot tuple 4 phan tu,
    moi phan tu duoc append vao dung cot tuong ung. Khi day, ngan se loai bo
    mot phan tu truoc khi them moi, theo chinh sach `pop`:
      - "min": loai mau co entropy (cot thu 4) nho nhat trong ngan
      - "max": loai mau co entropy lon nhat
      - gia tri khac: loai theo FIFO (phan tu cu nhat, chi so 0)

    Tham so `delay` (so lo) cho phep tre viec dua mot mau da co phan hoi
    (dung/sai) vao correct_mem/wrong_mem; mau duoc giu tam trong
    `delayed_queue` va chi thuc su chen vao sau khi du `delay` lo du lieu
    khong nhan ke tu luc duoc hoi -- mo phong do tre phan hoi thuc te cua
    oracle (xem --feedback_delay trong main.py).
    """

    def __init__(self, capacity, pop="min", delay=0):
        # feat, cls, domain, entropy
        self.correct_mem = [[], [], [], []]  # for correct samples
        self.wrong_mem = [[], [], [], []]  # for wrong samples
        self.u_mem = [[], [], [], []]  # for unlabeled samples : wait to be labeled
        self.capacity = capacity
        self.pop = pop
        self.delay = delay

        self.delayed_queue = []  # Queue for delayed insertions
        self.unlabeled_count = 0 # Count unlabeled samples

    def set_memory(self, state_dict):  # for tta_attack
        self.correct_mem = [ls[:] for ls in state_dict['correct_mem']]
        self.wrong_mem = [ls[:] for ls in state_dict['wrong_mem']]
        self.u_mem = [ls[:] for ls in state_dict['u_mem']]

        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['correct_mem'] = [ls[:] for ls in self.correct_mem]
        dic['wrong_mem'] = [ls[:] for ls in self.wrong_mem]
        dic['u_mem'] = [ls[:] for ls in self.u_mem]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        dic = {'correct_mem': self.correct_mem,
               'wrong_mem': self.wrong_mem,
               'u_mem': self.u_mem,
               'capacity': self.capacity}
        return dic

    def get_correct_memory(self):
        return self.correct_mem

    def get_wrong_memory(self):
        return self.wrong_mem

    def get_u_memory(self):
        return self.u_mem

    def get_occupancy(self, mem):
        return len(mem[0])  # need to be checked

    def add_instance(self, instance):
        raise NotImplementedError

    def add_correct_instance(self, instance):
        """Them mot mau (feat, cls, domain, entropy) da duoc oracle xac nhan
        dung. Neu `delay` > 0, mau duoc xep vao delayed_queue thay vi chen
        ngay vao correct_mem -- xem _process_delayed_queue()."""
        assert len(instance) == 4
        if self.delay > 0:
            self.delayed_queue.append(('correct', instance, self.unlabeled_count))
        else:
            self._insert_instance(self.correct_mem, instance)

    def add_wrong_instance(self, instance):
        """Giong add_correct_instance() nhung cho mau da duoc xac nhan sai,
        chen (co the tre) vao wrong_mem."""
        assert len(instance) == 4
        if self.delay > 0:
            self.delayed_queue.append(('wrong', instance, self.unlabeled_count))
        else:
            self._insert_instance(self.wrong_mem, instance)

    def add_u_instance(self, instance):
        """Them mot mau chua duoc hoi phan hoi vao u_mem (luon chen ngay,
        khong qua delayed_queue vi khong lien quan toi tre phan hoi). Sau
        moi --update_every_x mau duoc them (tuong duong 1 lo), kich hoat
        _process_delayed_queue() de giai phong cac mau dang cho du so lo
        tre quy dinh."""
        assert len(instance) == 4
        self.unlabeled_count += 1
        self._insert_instance(self.u_mem, instance)

        # After every 64 unlabeled samples (one batch), check delayed queue
        if self.unlabeled_count % conf.args.update_every_x == 0:
            self._process_delayed_queue()

    def _insert_instance(self, mem, instance):
        """Chen `instance` (4 phan tu) vao ngan `mem`, loai bot 1 phan tu
        theo chinh sach self.pop truoc neu ngan da day (self.capacity)."""
        if self.get_occupancy(mem) >= self.capacity:
            self.remove_instance(mem, pop=self.pop)
        for i, dim in enumerate(mem):
            dim.append(instance[i])

    def _process_delayed_queue(self):
        """Duyet delayed_queue, chuyen cac mau da cho du >= self.delay lo
        (tinh tu luc duoc dua vao hang doi) sang dung ngan correct_mem/
        wrong_mem, roi xoa chung khoi hang doi."""
        # Process delayed insertions if delay batches have passed
        ready_instances = []
        for idx, (mem_type, instance, unlabeled_start) in enumerate(self.delayed_queue):
            if (self.unlabeled_count - unlabeled_start) // conf.args.update_every_x >= self.delay:
                ready_instances.append(idx)
                if mem_type == 'correct':
                    self._insert_instance(self.correct_mem, instance)
                elif mem_type == 'wrong':
                    self._insert_instance(self.wrong_mem, instance)

        # Remove inserted instances from delayed queue
        for idx in reversed(ready_instances):
            self.delayed_queue.pop(idx)

    def remove_instance(self, mem, pop=None):
        """Loai bo 1 phan tu khoi ngan `mem` theo chinh sach `pop`: "min"/"max"
        loai mau co entropy (cot thu 4) nho nhat/lon nhat, con lai loai theo
        FIFO (phan tu chi so 0, tuc cu nhat)."""
        if pop == "min":
            target_idx = np.argmin(mem[3])
        elif pop == "max":
            target_idx = np.argmax(mem[3])
        else:
            target_idx = 0
        self.remove_instance_by_index(mem, target_idx)

    def remove_instance_by_index(self, mem, index):
        for dim in mem:
            dim.pop(index)
        return

    def remove_u_instance_by_index(self, index):
        self.remove_instance_by_index(self.u_mem, index)

    def reset(self):
        self.u_mem = [[], [], [], []]
        self.correct_mem = [[], [], [], []]
        self.wrong_mem = [[], [], [], []]
