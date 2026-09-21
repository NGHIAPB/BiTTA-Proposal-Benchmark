# CIFAR-10-C

Chua co du lieu trong thu muc nay (chi la scaffold).

## Nguon du lieu goc

- Anh nhiem (CIFAR-10-C, dinh dang `.npy`, 15 corruption x 5 severity): tai tu Zenodo
  — https://zenodo.org/record/2535967 — giai nen vao day (`dataset/CIFAR-10-C/`, cac
  file `<corruption>.npy` + `labels.npy` nam thang o thu muc goc, chua qua xu ly).
- Anh sach (CIFAR-10 goc, dinh dang pickle 5 `data_batch_*` + `test_batch`): tai tu
  https://www.cs.toronto.edu/~kriz/cifar.html, giai nen vao `dataset/cifar-10-batches-py/`.

## Xu ly ve dinh dang project dung

```
python src/process_cifar.py cifar-10c   # CHAY TU THU MUC GOC project (script doc dataset/CIFAR-10-C va dataset/cifar-10-batches-py theo os.getcwd())
```

Script se doc 2 nguon tren va sinh ra cau truc thu muc ma `src/data_loader/CIFAR10Dataset.py`
(va cac runner khac trong project) mong doi:

```
CIFAR-10-C/
├── origin/
│   ├── original.npy   # anh train goc (5 batch da gop)
│   └── labels.npy
└── corrupted/
    └── severity-<1..5>/
        ├── <corruption>.npy   # vd gaussian_noise.npy, tung file 10000 anh
        ├── test.npy            # anh test sach (khong bi nhiem), lap lai o moi severity
        └── labels.npy
```

## Checkpoint pretrain di kem

Checkpoint ResNet-18 (kieu CIFAR, `--model resnet18`) huan luyen tren CIFAR-10 sach
(`pretrained_weights/cifar10/cp_last_<seed>.pth.tar`) can duoc train rieng truoc khi chay TTA — xem
`pretrained_weights/README.md`.
