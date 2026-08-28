# PACS

Chua co du lieu trong thu muc nay (chi la scaffold).

## Cau truc thu muc can co (chuan torchvision.ImageFolder)
```
PACS/
├── art_painting/<7 lop: dog, elephant, giraffe, guitar, horse, house, person>/...jpg
├── cartoon/<7 lop>/...jpg
├── photo/<7 lop>/...jpg      (mien nguon)
└── sketch/<7 lop>/...jpg
```

## Nguon da xac nhan dung duoc
Ban DomainBed "kfold" (da loai anh trung lap giua cac domain) — vi du dataset
`tranductuankien/domain-generalization-dataset` tren Kaggle, duong dan con
`dataset/PACS/kfold/`.

## Checkpoint pretrained
Chi co **1 checkpoint dung chung ca 3/5 seed** (khac CIFAR/Tiny-ImageNet co checkpoint
rieng tung seed) — dat tai `../../pretrained_weights/pacs/cp_last.pth.tar`.
Neu chua co, can huan luyen source model qua `--method Src --tgt test` (xem
`src/main.py`).
