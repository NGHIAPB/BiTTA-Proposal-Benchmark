# TTA-Proposal-Benchmark

Benchmark test-time adaptation (TTA) cho đề án tốt nghiệp "Proposal cho BiTTA": so sánh **BiTTA-Proposal (C1 + C4)**
với BiTTA gốc, 3 phương pháp TTA khác được chạy ở chế độ **có binary feedback**, và 2 phương pháp **không dùng feedback**
(MEMO, DeYO), cùng một backbone **ResNet-18**.

## Kịch bản thực nghiệm

**Phương pháp** (7): TENT\*, EATA\*, SAR\*, BiTTA, BiTTA-Proposal (C1 + C4), MEMO, DeYO.
(\* = phiên bản có binary feedback `--enable_bitta`, tương ứng biến thể "B" trong Table 1 của paper BiTTA. MEMO và DeYO
**không** dùng binary feedback.)

| Setting | Dataset | Phương pháp | Model |
|---|---|---|---|
| **Continuous** | CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C, PACS | TENT\*, EATA\*, SAR\*, BiTTA, BiTTA-Proposal, **DeYO** | `resnet18` (CIFAR-C) · `resnet18_pretrained` (Tiny-ImageNet-C, PACS) |
| **Mixed shift** | CIFAR-10-C, PACS | như Continuous (gồm DeYO) | như trên |
| **Fully TTA** | CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C, PACS, ImageNet-R, ColoredMNIST, **WaterBirds** | **cả 7 phương pháp** (gồm MEMO, DeYO) | như trên; ImageNet-R, ColoredMNIST, WaterBirds: `resnet18_pretrained` |

- **MEMO chỉ chạy Fully TTA**: thuật toán episodic (nạp lại checkpoint gốc trước *mỗi* ảnh test) nên không có trạng thái để
  "liên tục" mang qua các domain — Continuous/Mixed không có nghĩa với MEMO.
- **WaterBirds chỉ chạy Fully TTA** (1 domain đích `test`), với cả 7 phương pháp; báo cáo thêm độ chính xác theo 4 nhóm
  (LL/LS/SL/SS), trung bình nhóm và nhóm tệ nhất như DeYO.
- **DeYO chạy cả 3 setting**, trên đúng các dataset của kịch bản (Continuous: 4, Mixed: 2, Fully: 7).

Số run (3 seed; PACS 5 seed): nhóm BiTTA (5 phương pháp) Continuous 70 + Mixed 40 + Fully 115 = **225**; DeYO 14 + 8 + 23 =
**45**; MEMO (Fully) **23** → tổng **293** run (MEMO/DeYO Fully mỗi run gồm nhiều corruption/domain chạy nối tiếp).
Model **không** phải trục độc lập: do dataset quyết định (`resnet18` là ResNet-18 kiểu CIFAR, chỉ hợp ảnh 32×32).

| Phương pháp | Nguồn |
|---|---|
| TENT | Wang et al., ICLR 2021 |
| EATA | Niu et al., ICML 2022 |
| SAR | Niu et al., ICLR 2023 |
| BiTTA | Lee et al., ICML 2025 |
| BiTTA-Proposal (C1 + C4) | Đóng góp của đề án |
| MEMO | Zhang et al., NeurIPS 2022 |
| DeYO | Lee et al., ICLR 2024 |

## Mục lục

- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị dữ liệu và checkpoint](#chuẩn-bị-dữ-liệu-và-checkpoint)
- [Cách chạy](#cách-chạy)
- [Ba setting được cài đặt như thế nào](#ba-setting-được-cài-đặt-như-thế-nào)
- [Ghi chú kỹ thuật](#ghi-chú-kỹ-thuật)
- [Mở rộng project](#mở-rộng-project)
- [License / nguồn gốc code](#license--nguồn-gốc-code)
- [Trạng thái kiểm chứng](#trạng-thái-kiểm-chứng)

## Cấu trúc project

```
TTA-Proposal-Benchmark/
├── configs/
│   ├── methods/                   # siêu tham số từng phương pháp (tent/eata/sar/bitta_*/deyo/memo.yaml)
│   ├── datasets/                  # thiết lập từng dataset (7 file .yaml)
│   └── settings/                  # 3 setting: continuous / mixed_shift / fully_tta
├── src/
│   ├── main.py                    # entrypoint duy nhất (TENT/EATA/SAR/BiTTA + huấn luyện nguồn --method Src)
│   ├── conf.py                    # cấu hình dataset (opt dict)
│   ├── process_cifar.py           # tiền xử lý CIFAR-10-C/100-C từ file .npy Zenodo
│   ├── learner/                   # dnn.py (lớp cơ sở + chọn mẫu C1/C4), tent.py, sar.py, eata.py, bitta.py
│   ├── models/ResNet.py           # ResNet-18 (kiểu CIFAR) và ResNetDropout18 (torchvision + Dropout cho MC-Dropout)
│   ├── data_loader/               # CIFAR10/100, TinyImageNet, PACS, ImageNet-R, ColoredMNIST, WaterBirds
│   ├── methods/deyo/              # DeYO (runner riêng, thuật toán từ repo tác giả; dataset/scenario_dataset.py nối 7 dataset)
│   ├── methods/memo/              # MEMO (runner riêng tự chứa, chỉ Fully TTA)
│   └── utils/                     # loss_functions, memory (FIFO), active_memory (ActivePriorityFIFO), calibration...
├── scripts/
│   ├── smoke_test.sh              # kiểm tra nhanh 3 setting (--nsample 500) trước khi chạy full
│   ├── run_bitta_family.sh        # 1 run: <method> <dataset> <setting> <seed>  (TENT/EATA/SAR/BiTTA/BiTTA-Proposal)
│   ├── run_deyo_family.sh         # DeYO: <method> <dataset> <setting> <seed>
│   ├── run_memo_family.sh         # MEMO (Fully): <method> <dataset> <corruption> <seed> [level]
│   ├── run_all_bitta_family.sh    # 225 run nhóm BiTTA (có lọc SETTINGS/DATASETS, SKIP_DONE)
│   └── run_all_other_methods.sh   # DeYO + MEMO (lọc WHICH/SETTINGS/DATASETS)
├── dataset/                       # (rỗng) CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C, imagenet-r, ColoredMNIST, WaterBirds
├── domainbed_dataset/             # (rỗng) PACS
├── pretrained_weights/            # (rỗng) checkpoint theo từng dataset
├── log/                           # output (online_eval.json)
├── results/aggregate_results.py   # tổng hợp bảng so sánh theo (dataset, setting)
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.9, PyTorch ≥ 2.0 với build CUDA phù hợp GPU. Cần GPU: `src/main.py` gọi `.cuda()` trực tiếp.

## Chuẩn bị dữ liệu và checkpoint

Mỗi thư mục con của `dataset/`, `domainbed_dataset/`, `pretrained_weights/` có `README.md` riêng (nguồn tải, cấu trúc
thư mục, cách tạo checkpoint). Tóm tắt:

| Dataset | Dữ liệu | Checkpoint |
|---|---|---|
| CIFAR-10-C / CIFAR-100-C | `dataset/CIFAR-10-C`, `dataset/CIFAR-100-C` | `pretrained_weights/cifar10\|cifar100/cp_last_<seed>.pth.tar` |
| Tiny-ImageNet-C | `dataset/Tiny-ImageNet-C` | `pretrained_weights/tiny-imagenet/cp_last_<seed>.pth.tar` |
| PACS | `domainbed_dataset/PACS` | `pretrained_weights/pacs/cp_last.pth.tar` (1 file chung) |
| ImageNet-R | `dataset/imagenet-r` | **không cần** (torchvision pretrained) |
| ColoredMNIST | `dataset/ColoredMNIST` (tự tạo từ MNIST) | `pretrained_weights/colored-mnist/cp_last_<seed>.pth.tar` (tự huấn luyện) |
| WaterBirds | `dataset/WaterBirds/waterbird_complete95_forest2water2/` (`metadata.csv` + ảnh, hoặc file `*.h5py`) | `pretrained_weights/waterbirds/cp_last.pth.tar` (1 file chung, tự huấn luyện `--method Src --src train`) |

Checkpoint được tạo bằng `src/main.py --method Src` — xem `pretrained_weights/README.md`.
**Cả 7 phương pháp dùng chung dữ liệu và checkpoint này** (DeYO qua `scenario_dataset.py`, MEMO qua `utils/data.py`), nên
không cần chuẩn bị riêng cho MEMO/DeYO.

## Cách chạy

### 1. Kiểm tra nhanh (bắt buộc trước khi chạy full)

```bash
bash scripts/smoke_test.sh
```

### 2. Một run

```bash
# bash scripts/run_bitta_family.sh <method> <dataset> <setting> <seed>
bash scripts/run_bitta_family.sh bitta_proposal cifar10_c continuous 0
bash scripts/run_bitta_family.sh tent           pacs       mixed      0
bash scripts/run_bitta_family.sh eata           imagenet_r fully      1
```

`method`: `tent | eata | sar | bitta_baseline | bitta_proposal` · `dataset`: `cifar10_c | cifar100_c | tiny_imagenet_c | pacs |
imagenet_r | colored_mnist | waterbirds` · `setting`: `continuous | mixed | fully`. Script từ chối tổ hợp không thuộc kịch bản.
Biến môi trường: `NSAMPLE` (giới hạn số ảnh; ImageNet-R mặc định 10000), `FISHER_SIZE` (EATA, mặc định 2000).

DeYO và MEMO có script riêng:

```bash
# DeYO: bash scripts/run_deyo_family.sh <method> <dataset> <setting> <seed>   (method = deyo)
bash scripts/run_deyo_family.sh deyo cifar10_c continuous 0
bash scripts/run_deyo_family.sh deyo pacs      mixed      0
bash scripts/run_deyo_family.sh deyo waterbirds fully     0     # Fully: lặp qua từng corruption/domain, mỗi lượt 1 tiến trình
# MEMO (chỉ Fully, từng corruption): bash scripts/run_memo_family.sh memo <dataset> <corruption> <seed> [level]
NSAMPLE=2000 bash scripts/run_memo_family.sh memo cifar10 gaussian_noise 0 5
bash scripts/run_memo_family.sh memo waterbirds test 0
```

### 3. Toàn bộ kịch bản

```bash
bash scripts/run_all_bitta_family.sh          # 225 run nhóm BiTTA
WHICH="deyo memo" bash scripts/run_all_other_methods.sh   # 45 run DeYO + 23 run MEMO
# hoặc từng phần, có thể chạy tiếp sau khi bị ngắt:
SETTINGS=continuous DATASETS=cifar10_c SKIP_DONE=1 bash scripts/run_all_bitta_family.sh
WHICH=memo DATASETS=waterbirds NSAMPLE=2000 bash scripts/run_all_other_methods.sh
```

### 4. Tổng hợp bảng so sánh

```bash
cd results
python aggregate_results.py --dataset cifar10_c --setting continuous
python aggregate_results.py --all
```

Mixed shift báo cáo accuracy tích luỹ tại 25/50/75/100% luồng dữ liệu (như Table 2 của paper BiTTA).

## Ba setting được cài đặt như thế nào

| Setting | Cách chạy trong `src/main.py` | Áp dụng |
|---|---|---|
| Continuous | `--tgt cont` — 1 tiến trình, 15 corruption (hoặc 3 domain PACS: art → cartoon → sketch) nối tiếp, **không reset** mô hình | 4 dataset có corruption/domain |
| Mixed shift | `--tgt cont --random_setting` — các corruption/domain **trộn ngẫu nhiên** trong 1 luồng | chỉ CIFAR-10-C, PACS (`data_loader` chỉ hỗ trợ danh sách nhiều domain cho 2 dataset này) |
| Fully TTA | `--tgt <unit>` — **mỗi corruption/domain là 1 tiến trình riêng**, khởi động từ mô hình nguồn | cả 7 dataset; ImageNet-R (`corrupt`), ColoredMNIST và WaterBirds (`test`) chỉ có 1 domain đích nên **chỉ** chạy setting này |

`src/main.py` từ chối các tổ hợp không hợp lệ (ImageNet-R/ColoredMNIST/WaterBirds với `--tgt cont`; `--random_setting` ngoài
CIFAR-10-C/PACS). Fully TTA dùng tiến trình riêng thay vì `--reset_every_corruption` vì cờ đó chỉ reset mô hình,
optimizer và bộ đệm Platt — không reset trạng thái nội bộ của EATA (`current_model_probs`) và SAR.

**DeYO** (`src/methods/deyo/main.py`): Continuous = `--continual True` (mô hình/optimizer giữ nguyên qua các corruption);
Mixed = `--exp_type mix_shifts` (các domain nối thành 1 luồng); Fully = `--continual False --unit <unit>` (mỗi unit khởi
động lại từ checkpoint). **MEMO** chỉ Fully (`run_memo_family.sh`, mỗi ảnh nạp lại checkpoint).

## Ghi chú kỹ thuật

### TENT\*, EATA\*, SAR\* (có binary feedback)

Chạy với `--enable_bitta`: ngoài loss gốc, mỗi batch lấy `n_active_sample` (=3) mẫu để hỏi oracle (chọn ngẫu nhiên,
`--sample_selection random` mặc định), rồi cộng thêm loss `CE` cho mẫu đúng và `complement-CE` cho mẫu sai
(`DNN.get_bitta_ssl_loss`) — cách paper BiTTA điều chỉnh các baseline. BiTTA và BiTTA-Proposal dùng
`ActivePriorityFIFO` + MC-Dropout; TENT/EATA/SAR **bắt buộc** `--memory_type FIFO` (các learner
`assert isinstance(self.mem, FIFO)`). Siêu tham số từng method theo `tta.sh` của repo BiTTA gốc.
Không có kết quả "không feedback" của TENT/EATA/SAR trong kịch bản này.

### Siêu tham số chưa có trong paper (chưa tinh chỉnh)

| Dataset | BiTTA / BiTTA-Proposal | EATA |
|---|---|---|
| ImageNet-R | mượn Tiny-ImageNet-C (`epoch 5, lr 5e-5, dropout 0.1, n_dropouts 2, restoration 0.01`) | `e_margin = 0.4·ln(200)`, `lr 2.5e-4`, `d_margin 0.05`, `fisher_alpha 2000` |
| ColoredMNIST | mượn CIFAR (`epoch 3, lr 1e-4, dropout 0.3, n_dropouts 4`) | `e_margin = 0.4·ln(2)`, `lr 5e-3`, `d_margin 0.4`, `fisher_alpha 1` |

Paper BiTTA chỉ báo cáo 1 con số/dataset cho ImageNet-R và ColoredMNIST (Table 7, Appendix C) và không nêu cấu hình chi
tiết. Coi kết quả là **xu hướng**, không phải tái lập số tuyệt đối.

### ImageNet-R và ColoredMNIST

- ImageNet-R: `--nsample` giới hạn số ảnh nạp vào RAM (lấy ngẫu nhiên theo `--seed`, giữ thứ tự gốc): `main.py` giữ toàn bộ
  tensor float32 3×224×224 (~0,6 MB/ảnh), 30.000 ảnh ≈ 18 GB (đỉnh gấp đôi khi `torch.stack`).
- ColoredMNIST: loader tự tạo `train1.pt`/`train2.pt`/`test.pt` từ MNIST ở lần chạy đầu (cần internet) và đọc
  `.pt` bằng `torch.load(..., weights_only=False)` (bắt buộc với PyTorch ≥ 2.6). Nguồn `all_train` chỉ được nạp khi huấn
  luyện checkpoint (`--method Src`).

### MEMO, DeYO và WaterBirds

- **DeYO — sửa so với repo tác giả** (chỉ đấu nối/sửa lỗi, không đổi thuật toán `methods/*.py`): (1) bản gốc dựng lại mô
  hình từ checkpoint ở *mỗi* vòng corruption dù `--continual True`, nên "Continuous" thực chất chạy như Fully TTA — nay giữ
  mô hình/optimizer qua các corruption (EATA tính Fisher 1 lần ở corruption đầu); (2) thêm `--unit` cho Fully; (3) 7 dataset
  của project được nối qua `dataset/scenario_dataset.py` (`--dset CIFAR10-C | CIFAR100-C | TinyImageNet-C | PACS-scenario |
  ImageNetR-scenario | WaterBirds-scenario | ColoredMNIST-scenario`, `--model resnet18_scenario`) để dùng chung dữ liệu,
  checkpoint và lớp chuẩn hoá với nhóm BiTTA. Siêu tham số DeYO theo paper/repo gốc, **chưa tinh chỉnh** cho các dataset ngoài
  ImageNet-C/Waterbirds/ColoredMNIST (paper DeYO không báo cáo). WaterBirds dùng `lr_mul 5` như DeYO gốc; ResNet-18 chưa được
  paper DeYO kiểm chứng (paper dùng ResNet-50).
- **MEMO**: 32 augmix/ảnh + 1 bước SGD rồi nạp lại checkpoint → rất chậm. Tăng tốc mà giữ đúng thuật toán: `PROFILE=<n>`
  đo thời gian (augmix CPU / forward-backward GPU), `SHARDS=<k>` chia ảnh thành k phần chạy song song trên 1 GPU rồi gộp
  (mỗi ảnh độc lập nên kết quả tương đương); `NSAMPLE` chỉ dùng khi bắt buộc (đổi giao thức, phải ghi rõ). CIFAR dùng
  ResNet-18 kiểu CIFAR (conv1 3×3), đúng kiến trúc checkpoint.
- **WaterBirds** (Fully TTA, 7 phương pháp): chỉ số chính = trung bình 4 nhóm (`avg_group`) và nhóm tệ nhất (`worst_group`);
  nhóm BiTTA ghi `waterbirds_groups.json` cạnh `online_eval.json`. Cấu hình BiTTA/EATA mượn Tiny-ImageNet-C (chưa tinh chỉnh).

### Cache

`target_train_set` được cache ở `./cached_data/` với khoá **không chứa `--nsample`**. Sau khi đổi `--nsample` (ví dụ
smoke test rồi chạy full) hãy xoá `cached_data/<dataset>_*`. Với `--tgt cont`, cache của từng corruption được xoá ngay
sau khi chạy xong để tránh đầy ổ.

### Định dạng log

`log/<dataset>/<TENT|EATA|SAR|BiTTA>/tgt_<...>/<log_prefix>/.../online_eval.json`, với `log_prefix =
<method>_<dataset>_<setting>_s<seed>`; Continuous/Mixed nằm dưới `tgt_cont/` (Mixed: thư mục con `random/`), Fully nằm dưới
`tgt_<unit>/`. Thư mục `<dataset>` là giá trị `--dataset` của `main.py`: `cifar10`, `cifar100`, `tiny-imagenet`, `pacs`,
`imagenetR`, `colored-mnist`, `waterbirds`.

DeYO và MEMO ghi log văn bản (dòng `Result under <unit>. ... top1|average: X`): `log/deyo_<method>_<dataset>_<setting>_s<seed>/*.txt`
và `log/memo_<method>_<dataset>_s<seed>/<dataset>_<unit>_L<level>_s<seed>.txt`; `results/aggregate_results.py` đọc cả hai.

## Mở rộng project

### Thêm dataset mới

1. Thêm entry `opt` vào `src/conf.py` (theo mẫu `CIFAR10Opt`/`PACSOpt`).
2. Viết class Dataset trong `src/data_loader/` trả về bộ ba `(ảnh, nhãn, nhãn_domain)`; đăng ký nhánh trong
   `data_loader.domain_data_loader`.
3. Đăng ký nhánh `elif '<tên>' in conf.args.dataset:` trong `src/main.py` (chọn `opt`).
4. Thêm nhánh cho EATA (tính Fisher) trong `learner/eata.py` và lớp chuẩn hoá trong `utils/normalize_layer.py`.
5. Thêm ca tương ứng trong `scripts/run_bitta_family.sh` và `results/aggregate_results.py`.

### Model

Chỉ dùng ResNet-18: `resnet18` (`ResNet.ResNet18`, kiểu CIFAR, train từ đầu) và `resnet18_pretrained` (`ResNetDropout18` —
torchvision ResNet-18 + Dropout sau mỗi residual block cho MC-Dropout).

## License / nguồn gốc code

`src/*` (trừ `methods/`) là repo BiTTA gốc của đề án đã được rút gọn theo kịch bản (TENT/EATA/SAR/BiTTA, 7 dataset, ResNet-18),
cộng phần chọn mẫu C1 + C4 của Proposal trong `learner/dnn.py`; không chỉnh sửa logic thuật toán của các phương pháp.
`src/methods/deyo/` là code của tác giả DeYO (giữ nguyên thuật toán, LICENSE kèm theo; các chỗ sửa/đấu nối nêu ở
[MEMO, DeYO và WaterBirds](#memo-deyo-và-waterbirds)); `src/methods/memo/` là bản cài đặt MEMO tự chứa theo paper.

## Trạng thái kiểm chứng

Đã chạy thử **trên CPU với dữ liệu giả** (ảnh ngẫu nhiên, đúng cấu trúc thư mục của loader) và checkpoint tự huấn luyện
bằng `--method Src`. Chưa chạy trên dữ liệu và GPU thật, nên **chưa có số accuracy nào**; luồng chạy, đường dẫn log và
tổng hợp kết quả đã được kiểm tra — gồm DeYO (Continuous có giữ trạng thái, Mixed, Fully; 7 dataset), MEMO (cifar10,
ImageNet-R, ColoredMNIST, WaterBirds) và WaterBirds với cả nhóm BiTTA.
