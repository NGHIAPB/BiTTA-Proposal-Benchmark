# Pretrained weights

Chua co checkpoint trong cac thu muc con (chi la scaffold, moi thu muc rong).

| Thu muc | Ten file can co | Ghi chu |
|---|---|---|
| `cifar10/` | `cp_last_0.pth.tar`, `cp_last_1.pth.tar`, `cp_last_2.pth.tar` | 1 checkpoint/seed |
| `cifar100/` | tuong tu cifar10 | |
| `tiny-imagenet/` | tuong tu cifar10 | model=`resnet18_pretrained` |
| `pacs/` | `cp_last.pth.tar` (chi 1 file, dung chung moi seed) | |
| `waterbirds/` | huan luyen qua `src/methods/deyo/pretrain_waterbirds.py` | |

## Cách tạo checkpoint nếu chưa có

Dùng chính `src/main.py` với `--method Src` (huấn luyện offline trên tập nguồn sạch,
không TTA — dùng class `DNN` gốc):

```bash
cd src
python main.py --gpu_idx 0 --dataset cifar10 --method Src --tgt test \
    --model resnet18 --epoch 200 --seed 0 \
    --log_name ../log/ --log_prefix pretrain_seed0
```

Checkpoint được ghi ra **không** trực tiếp vào `pretrained_weights/`, mà theo quy ước
đường dẫn chung của `src/main.py` (`get_path()`):

```
log/cifar10/Src/tgt_test/pretrain_seed0/cp/cp_last.pth.tar
```

Cần copy/đổi tên file này vào đúng vị trí mà các script TTA (`run_bitta_family.sh`,
`run_memo_family.sh`) mong đợi, ví dụ:

```bash
cp log/cifar10/Src/tgt_test/pretrain_seed0/cp/cp_last.pth.tar \
   pretrained_weights/cifar10/cp_last_0.pth.tar
```

Lặp lại cho từng `--seed` (0/1/2) và từng dataset — đổi `--dataset`, `--model` (dùng
`resnet18_pretrained` cho `tiny-imagenet`/`pacs`) và `--epoch`/`--lr` theo đúng bảng
tham số trong `configs/methods/bitta_baseline.yaml`. Riêng `waterbirds/` dùng script
huấn luyện riêng: `src/methods/deyo/pretrain_waterbirds.py`.
