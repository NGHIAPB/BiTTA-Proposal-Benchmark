# Tiny-ImageNet-C

Chua co du lieu trong thu muc nay (chi la scaffold).

## Cau truc thu muc can co
```
Tiny-ImageNet-C/
├── origin/Data/train/<200 thu muc lop wnid>/...jpg   (anh sach, mien nguon)
└── corrupted/<ten_corruption>/<severity 1-5>/<wnid>/...jpg
```

## Nguon dataset tren Kaggle da xac nhan dung duoc
- Anh sach: `akash2sharma/tiny-imagenet` (giu nguyen cau truc `tiny-imagenet-200/train/`)
- Anh nhiem: `luckyhathaway/tiny-imagenet-c`

## Luu y quan trong
- Anh 224x224 -> can `--n_dropouts 2` (khong phai 4 mac dinh) de tranh CUDA OOM tren
  GPU ca nhan (da xac nhan qua thuc nghiem trong de an).
- Neu chay tren Kaggle: cache trung gian cua target_train_set co the lam day dia
  20GB working -- xem co che tu dong don cache da cai san trong `src/main.py`
  (chi kich hoat khi `conf.args.tgt == "cont"`).
