# Dataset loader WaterBirds cho ho BiTTA-family (TENT / EATA / SAR / BiTTA / BiTTA-Proposal).
#
# WaterBirds (Sagawa et al., ICLR 2020, "Distributionally Robust Neural Networks") la benchmark
# tuong quan gia: nhan y = {0: landbird, 1: waterbird}, nen anh place = {0: land, 1: water}.
# Ba split theo metadata.csv goc: 0 = train, 1 = val, 2 = test. O split test, tuong quan
# (y, place) bi dao nguoc/can bang so voi train -> mo hinh "an gian" bam vao nen se sai.
#
# Giao dien trung voi cac dataset khac cua ho BiTTA (PACSDataset, TINYIMAGENETDataset...):
#   WaterbirdsDataset(file, domains, max_source, transform)  ->  __getitem__ = (img, y, place)
# Phan tu thu 3 (o cac dataset khac la "domain label") o day la `place`, de tinh duoc do chinh
# xac theo 4 nhom (LL/LS/SL/SS) giong cach DeYO bao cao: xem compute_group_metrics().
#
# Hai backend du lieu (tu dong chon, uu tien h5py):
#   1. File .h5py do src/methods/deyo/pretrain_waterbirds.py::make_dataset() tao ra
#      (Waterbirds/<split>/<idx> = anh 224x224 uint8, attrs: y, place) -> dung chung voi DeYO.
#   2. Thu muc goc cua bo du lieu (waterbird_complete95_forest2water2) co metadata.csv +
#      anh .jpg -> doc truc tiep, resize (224, 224) giong make_dataset().
#
# Chuan hoa (mean/std ImageNet) KHONG nam trong transform ma nam trong NormalizeLayer cua
# mang (utils/normalize_layer.py, dataset 'waterbirds'), giong TinyImageNet/PACS -- de checkpoint
# state_dict (pretrained_weights/waterbirds/cp_last.pth.tar) dung chung voi MEMO.
import csv
import glob
import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

import conf

SPLIT_ID = {'train': 0, 'val': 1, 'test': 2}
IMG_SIZE = 224
H5_GROUP = 'Waterbirds'
H5_EXTS = ('*.h5py', '*.h5', '*.hdf5')
# group = 2 * y + place  (y: 0 landbird / 1 waterbird ; place: 0 land / 1 water)
GROUP_NAMES = ('LL', 'LS', 'SL', 'SS')  # landbird-land, landbird-sea, seabird-land, seabird-sea


def _root_candidates(root):
    """Ten thu muc tren Linux phan biet hoa/thuong: WaterBirds (repo nay) vs Waterbirds (DeYO)."""
    root = os.path.normpath(root)
    parent, base = os.path.split(root)
    cands = [root]
    for alt in ('WaterBirds', 'Waterbirds', 'waterbirds'):
        c = os.path.join(parent, alt)
        if c not in cands:
            cands.append(c)
    return cands


def _find_backend(root):
    """Tra ve (kind, path): ('h5', file.h5py) hoac ('csv', metadata.csv). Tim o root va 1 cap con."""
    for base in _root_candidates(root):
        if not os.path.isdir(base):
            continue
        search_dirs = [base] + sorted(d for d in glob.glob(os.path.join(base, '*')) if os.path.isdir(d))
        for d in search_dirs:
            for pat in H5_EXTS:
                hits = sorted(glob.glob(os.path.join(d, pat)))
                if hits:
                    return 'h5', hits[0]
        for d in search_dirs:
            meta = os.path.join(d, 'metadata.csv')
            if os.path.isfile(meta):
                return 'csv', meta
    raise FileNotFoundError(
        "Khong tim thay du lieu WaterBirds trong '%s' (thu ca %s). Can 1 trong 2: (a) file *.h5py "
        "tao boi src/methods/deyo/pretrain_waterbirds.py, hoac (b) thu muc chua metadata.csv + anh. "
        "Xem dataset/WaterBirds/README.md." % (root, [os.path.basename(c) for c in _root_candidates(root)]))


class WaterbirdsDataset(torch.utils.data.Dataset):

    def __init__(self, file='', domains=None, max_source=100, transform='val', max_samples=None):
        """
        file:        thu muc du lieu (mac dinh conf.WATERBIRDSOpt['file_path']).
        domains:     list ten split, ['train'] | ['val'] | ['test'] (chi lay phan tu dau).
        transform:   'src' (huan luyen: lat ngang + ToTensor) | 'val' (chi ToTensor).
        max_samples: neu dat va nho hon so anh cua split -> chi giu ngau nhien (seed = --seed)
                     dung chinh so anh nay, giu nguyen thu tu goc (cho smoke test / may it RAM).
        """
        domains = ['test'] if not domains else list(domains)
        if isinstance(domains, str):
            domains = [domains]
        split = domains[0]
        if split not in SPLIT_ID:
            raise ValueError("Split WaterBirds khong hop le: %r (chi co %s)" % (split, list(SPLIT_ID)))
        self.split = split
        self.domain = split
        self.max_source = max_source
        self.transform_type = transform
        root = file if file else conf.WATERBIRDSOpt['file_path']
        self.kind, self.path = _find_backend(root)

        if transform == 'src':
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        elif transform == 'val':
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            raise NotImplementedError(transform)

        self._h5 = None
        if self.kind == 'h5':
            self._index_h5()
        else:
            self._index_csv()

        if max_samples is not None and 0 < max_samples < len(self.labels):
            seed = getattr(getattr(conf, 'args', None), 'seed', 0) or 0
            if getattr(getattr(conf, 'args', None), 'tgt_train_dist', 1) == 0:  # "real order": n anh dau
                keep = np.arange(max_samples)
            else:  # ngau nhien co seed, giu thu tu goc (target_data_processing se xao lai theo dist)
                keep = np.sort(np.random.RandomState(seed).choice(len(self.labels), max_samples, replace=False))
            self._apply_subset(keep)
        else:
            self._items = list(range(len(self.labels)))

    # ---------------------------------------------------------------- indexing
    def _index_h5(self):
        import h5py  # import tre: chi can khi dung backend h5py
        with h5py.File(self.path, 'r') as f:
            grp = f[H5_GROUP][self.split]
            n = len(grp)
            self.labels = np.empty(n, dtype=np.int64)
            self.places = np.empty(n, dtype=np.int64)
            for i in range(n):
                attrs = grp[str(i)].attrs
                self.labels[i] = int(attrs['y'])
                self.places[i] = int(attrs['place'])

    def _index_csv(self):
        base = os.path.dirname(self.path)
        want = SPLIT_ID[self.split]
        files, labels, places = [], [], []
        with open(self.path, 'r', newline='') as fh:
            rdr = csv.reader(fh)
            next(rdr)  # header: img_id,img_filename,y,split,place,place_filename
            for row in rdr:
                if int(row[3]) != want:
                    continue
                files.append(os.path.join(base, row[1]))
                labels.append(int(row[2]))
                places.append(int(row[4]))
        self._files = files
        self.labels = np.asarray(labels, dtype=np.int64)
        self.places = np.asarray(places, dtype=np.int64)

    def _apply_subset(self, keep):
        self.labels = self.labels[keep]
        self.places = self.places[keep]
        if self.kind == 'csv':
            self._files = [self._files[i] for i in keep]
            self._items = list(range(len(self.labels)))
        else:
            self._items = [int(i) for i in keep]  # chi so trong split h5py

    # ---------------------------------------------------------------- Dataset API
    def __len__(self):
        return len(self.labels)

    def get_num_domains(self):
        return 1

    def _load_h5(self, i):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.path, 'r')
        arr = self._h5[H5_GROUP][self.split][str(self._items[i])][()]
        return Image.fromarray(np.asarray(arr, dtype=np.uint8))

    def _load_csv(self, i):
        return Image.open(self._files[i]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if idx < 0 or idx >= len(self.labels):
            raise IndexError(idx)
        img = self._load_h5(idx) if self.kind == 'h5' else self._load_csv(idx)
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE))
        img = self.transform(img)
        return img, int(self.labels[idx]), torch.tensor(int(self.places[idx]))

    # h5py.File khong pickle duoc -> bo handle khi pickle/fork DataLoader worker
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def __del__(self):
        try:
            if getattr(self, '_h5', None) is not None:
                self._h5.close()
        except Exception:
            pass


def compute_group_metrics(gt, pred, place):
    """Do chinh xac (%) theo 4 nhom, dung cong thuc cua DeYO: avg = trung binh 4 nhom.

    gt/pred/place: chuoi so nguyen cung do dai. Nhom = 2 * y + place.
    """
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    place = np.asarray(place, dtype=np.int64)
    assert gt.shape == pred.shape == place.shape, "gt/pred/place phai cung do dai"
    group = 2 * gt + place
    out = {}
    accs = []
    for g, name in enumerate(GROUP_NAMES):
        m = group == g
        out['n_' + name] = int(m.sum())
        if m.any():
            out[name] = 100.0 * float((pred[m] == gt[m]).mean())
            accs.append(out[name])
        else:
            out[name] = float('nan')
    out['avg_group'] = float(np.mean(accs)) if accs else float('nan')
    out['worst_group'] = float(np.min(accs)) if accs else float('nan')
    out['raw_acc'] = 100.0 * float((pred == gt).mean()) if len(gt) else float('nan')
    out['n_total'] = int(len(gt))
    return out


def dump_group_metrics(learner, out_dir):
    """Ghi waterbirds_groups.json canh online_eval.json (goi sau learner.dump_eval_online_result()).

    Thu tu json_eval['gt'/'pred'] trung thu tu target_train_set (train_online xu ly tuan tu),
    nen place = target_train_set[2][:len(gt)]. Khong bao gio lam hong luot chay: moi loi chi canh bao.
    """
    import json
    try:
        gt = learner.json_eval['gt']
        pred = learner.json_eval['pred']
        tgt_cls = learner.target_train_set[1]
        place = learner.target_train_set[2]
        n = len(gt)
        place = [int(p) for p in place[:n]]
        chk = [int(c) for c in tgt_cls[:n]]
        if chk != [int(g) for g in gt]:
            print('[WaterBirds] CANH BAO: thu tu gt khac target_train_set -> bo qua thong ke nhom')
            return None
        res = compute_group_metrics(gt, pred, place)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'waterbirds_groups.json'), 'w') as fh:
            json.dump(res, fh, indent=2)
        print('[WaterBirds] ' + ', '.join('%s=%.2f' % (k, res[k]) for k in GROUP_NAMES) +
              ' | avg_group=%.2f worst_group=%.2f raw_acc=%.2f' % (res['avg_group'], res['worst_group'], res['raw_acc']))
        return res
    except Exception as e:  # pragma: no cover
        print('[WaterBirds] CANH BAO: khong ghi duoc thong ke nhom: %r' % (e,))
        return None
