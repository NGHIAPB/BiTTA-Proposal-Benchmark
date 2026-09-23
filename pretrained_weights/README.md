# Pretrained weights

Chưa có checkpoint trong các thư mục con (chỉ là scaffold, mỗi thư mục rỗng). Đường dẫn/tên file dưới đây là
đúng những gì `scripts/run_bitta_family.sh` mong đợi.

| Thư mục | Tên file cần có | Model | Ghi chú |
|---|---|---|---|
| `cifar10/` | `cp_last_0.pth.tar`, `cp_last_1.pth.tar`, `cp_last_2.pth.tar` | `resnet18` (kiểu CIFAR, train từ đầu) | 1 checkpoint/seed |
| `cifar100/` | tương tự cifar10 | `resnet18` | |
| `tiny-imagenet/` | tương tự cifar10 | `resnet18_pretrained` (khởi tạo ImageNet, fine-tune trên nguồn) | |
| `pacs/` | `cp_last.pth.tar` (chỉ 1 file, dùng chung mọi seed) | `resnet18_pretrained` | |
| `colored-mnist/` | `cp_last_0.pth.tar`, `cp_last_1.pth.tar`, `cp_last_2.pth.tar` | `resnet18_pretrained` | tự huấn luyện, xem `dataset/ColoredMNIST/README.md` |
| `waterbirds/` | `cp_last.pth.tar` (chỉ 1 file, dùng chung mọi seed) | `resnet18_pretrained` | tự huấn luyện, xem `dataset/WaterBirds/README.md` |
| (ImageNet-R) | **không cần file** | `resnet18_pretrained` (torchvision tự tải, lọc 1000→200 logit) | |

## Cách tạo checkpoint

Dùng chính `src/main.py` với `--method Src` (huấn luyện offline trên miền nguồn sạch, không TTA). **Chạy từ thư mục
gốc project** (không `cd src`) — `conf.py` dùng đường dẫn tương đối (`./dataset/...`) tính theo thư mục làm việc hiện
tại, giống hệt cách `scripts/run_bitta_family.sh` gọi `python src/main.py`:

```bash
python src/main.py --gpu_idx 0 --dataset cifar10 --method Src --src original --tgt gaussian_noise-5 \
    --model resnet18 --epoch 200 --seed 0 \
    --log_name log/ --log_prefix pretrain_seed0
```

Checkpoint được ghi **không** trực tiếp vào `pretrained_weights/`, mà theo quy ước đường dẫn của `src/main.py`
(`get_path()`):

```
log/cifar10/Src/src_original/tgt_gaussian_noise-5/pretrain_seed0/cp/cp_last.pth.tar
```

Copy/đổi tên vào đúng vị trí mà script TTA mong đợi:

```bash
cp log/cifar10/Src/src_original/tgt_gaussian_noise-5/pretrain_seed0/cp/cp_last.pth.tar \
   pretrained_weights/cifar10/cp_last_0.pth.tar
```

Lặp lại cho từng `--seed` (0/1/2). Tham số `--src`, `--tgt`, `--model` theo từng dataset:

| Dataset | `--dataset` | `--src` | `--tgt` (chỉ để đánh giá cuối) | `--model` |
|---|---|---|---|---|
| CIFAR-10-C | `cifar10` | `original` | `gaussian_noise-5` | `resnet18` |
| CIFAR-100-C | `cifar100` | `original` | `gaussian_noise-5` | `resnet18` |
| Tiny-ImageNet-C | `tiny-imagenet` | `original` | `gaussian_noise-5` | `resnet18_pretrained` |
| PACS | `pacs` | `photo` | `sketch` | `resnet18_pretrained` |
| ColoredMNIST | `colored-mnist` | `all_train` | `test` | `resnet18_pretrained` |
| WaterBirds | `waterbirds` | `train` | `test` | `resnet18_pretrained` |

Lưu ý:
- **Không truyền `--lr`** khi huấn luyện nguồn: `main.py` chia `lr` cho 64 khi `--memory_size 1` (mặc định); dùng `lr`
  mặc định trong `src/conf.py` cho từng dataset.
- `--epoch` chỉ là ví dụ — chọn theo độ chính xác trên miền nguồn/validation. Số epoch và `lr` dùng cho checkpoint
  gốc của paper BiTTA nằm trong `train_src.sh` của repo gốc.
- Các dataset ảnh 224×224 (`tiny-imagenet`, `pacs`) tốn nhiều VRAM khi huấn luyện; giảm `--dataloader_batch_size` nếu OOM.
