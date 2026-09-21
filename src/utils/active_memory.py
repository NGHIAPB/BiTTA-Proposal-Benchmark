import random

import numpy as np
import torch

import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator

class ActivePriorityFIFO:
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
        assert len(instance) == 4
        if self.delay > 0:
            self.delayed_queue.append(('correct', instance, self.unlabeled_count))
        else:
            self._insert_instance(self.correct_mem, instance)

    def add_wrong_instance(self, instance):
        assert len(instance) == 4
        if self.delay > 0:
            self.delayed_queue.append(('wrong', instance, self.unlabeled_count))
        else:
            self._insert_instance(self.wrong_mem, instance)

    def add_u_instance(self, instance):
        assert len(instance) == 4
        self.unlabeled_count += 1
        self._insert_instance(self.u_mem, instance)

        # After every 64 unlabeled samples (one batch), check delayed queue
        if self.unlabeled_count % conf.args.update_every_x == 0:
            self._process_delayed_queue()

    def _insert_instance(self, mem, instance):
        if self.get_occupancy(mem) >= self.capacity:
            self.remove_instance(mem, pop=self.pop)
        for i, dim in enumerate(mem):
            dim.append(instance[i])

    def _process_delayed_queue(self):
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
