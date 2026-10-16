import conf  # noqa: F401  (giu thu tu import nhu ban goc)


class FIFO:
    """Bo nho hang doi vao-truoc-ra-truoc don gian, dung lam bo dem lo cho
    TENT/EATA/SAR (cac phuong phap khong co phan hoi nhi phan nen chi can
    mot ngan duy nhat, khac voi ActivePriorityFIFO 3 ngan cua BiTTA).

    self.data la 3 "cot" song song [feat, cls, domain]; mot instance them
    vao la mot tuple 3 phan tu, moi phan tu duoc append vao dung cot. Khi
    day (dat self.capacity), phan tu cu nhat (chi so 0) bi loai truoc khi
    them moi -- xem add_instance()/remove_instance().
    """

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
        """Them mot mau (feat, cls, domain), loai mau cu nhat truoc neu bo
        nho da day (self.capacity)."""
        assert (len(instance) == 3)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        """Loai mau cu nhat (chi so 0) khoi ca 3 cot, dung FIFO."""
        for dim in self.data:
            dim.pop(0)
        pass

    def remove_instance_by_index(self, index):
        """Loai mau o vi tri `index` bat ky (khong nhat thiet la mau cu
        nhat) khoi ca 3 cot."""
        new_data = []
        for dim in self.data:
            new_data += [dim[:index] + dim[index+1 : ]]
        self.data = new_data
        return

    def reset(self):
        self.data = [[], [], []]
