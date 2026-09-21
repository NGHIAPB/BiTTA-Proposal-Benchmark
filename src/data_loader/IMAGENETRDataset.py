# ImageNet-R (Hendrycks et al., 2021): 30.000 anh "rendition" (tranh ve, hoat hoa, dieu khac...) cua 200 lop
# ImageNet. Chi co 1 domain dich duy nhat ("corrupt") -> chi chay duoc Fully TTA (khong co continuous/mixed).
#
# Cau truc thu muc: dataset/imagenet-r/<200 thu muc wnid>/*.jpg   (ban tai chinh thuc imagenet-r.tar)
# Nhan = thu tu sap xep cua 200 ten thu muc; mo hinh ResNet-18 pretrained 1000 lop duoc loc ve 200 lop bang
# conf.IMAGENET_R['indices_in_1k'] (ResNetDropout(filter=...), xem learner/dnn.py) -> khong can checkpoint.
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import conf

opt = conf.IMAGENET_R


class ImageNetRDataset(torch.utils.data.Dataset):

    def __init__(self, file='',
                 domain=None, activities=None,
                 max_source=100, transform='none', max_samples=None):
        self.domain = domain
        self.activity = activities
        self.max_source = max_source
        self.max_samples = max_samples
        self.file_path = opt['file_path']
        self.transform_type = transform

        assert domain == "corrupt", "ImageNet-R chi co 1 domain dich: 'corrupt' (nhan %r)" % (domain,)

        if transform == 'src':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        elif transform == 'val':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        else:
            raise NotImplementedError

        self.dataset = ImageFolder(self.file_path)
        n = len(self.dataset.samples)
        if self.max_samples is not None and 0 < self.max_samples < n:
            # target_data_processing giu TOAN BO tensor float32 3x224x224 (~0.6 MB/anh) trong RAM
            # -> chi giu max_samples anh (giu thu tu goc de --tgt_train_dist xu ly tiep).
            if getattr(conf.args, 'tgt_train_dist', 1) == 0:  # "real order": n anh dau
                keep = np.arange(self.max_samples)
            else:  # ngau nhien theo --seed
                seed = getattr(conf.args, 'seed', 0) or 0
                keep = np.sort(np.random.RandomState(seed).choice(n, self.max_samples, replace=False))
            self.dataset.samples = [self.dataset.samples[i] for i in keep]
            self.dataset.targets = [self.dataset.targets[i] for i in keep]
            self.dataset.imgs = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return 1

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, torch.tensor(0)
