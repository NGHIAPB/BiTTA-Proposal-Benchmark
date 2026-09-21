import conf  # noqa: F401  (giu thu tu import nhu ban goc)


class FIFO:
    def __init__(self, capacity):
        self.data = [[], [], []]  # feat, cls, domain
        self.capacity = capacity
        pass

    def set_memory(self, state_dict):  # for tta_attack
        self.data = [ls[:] for ls in state_dict['data']]
        if 'capacity' in state_dict.keys():
            self.capacity = state_dict['capacity']

    def save_state_dict(self):
        dic = {}
        dic['data'] = [ls[:] for ls in self.data]
        dic['capacity'] = self.capacity
        return dic

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

    def remove_instance_by_index(self, index):
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1 : ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], []]
