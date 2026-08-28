# TTA-Proposal-Benchmark

Benchmark test-time adaptation (TTA) cho đề án tốt nghiệp "Proposal cho BiTTA", so sánh
bảy phương pháp trên cùng một bộ dataset và cùng một backbone **ResNet-18**:

| # | Phương pháp | Nguồn |
|---|---|---|
| 1 | TENT | Wang et al., ICLR 2021 |
| 2 | EATA | Niu et al., ICML 2022 |
| 3 | SAR | Niu et al., ICLR 2023 |
| 4 | DeYO | Lee et al., ICLR 2024 |
| 5 | MEMO | Zhang, Levine & Finn, NeurIPS 2022 |
| 6 | BiTTA-baseline | Lee et al., ICML 2025 |
| 7 | BiTTA-Proposal (C1 + C4) | Đóng góp của đề án |

Kiến trúc thư mục tham khảo repo **Benchmark-TTA**: tách `configs/` theo method/dataset,
`scripts/` cho từng kịch bản thực nghiệm, `results/` cho lớp tổng hợp bảng so sánh.

## Mục lục

- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Cách chạy](#cách-chạy)
- [3 setting TTA](#3-setting-tta-mở-rộng-được)
- [Kiến trúc: vì sao 3 entrypoint riêng biệt](#kiến-trúc-vì-sao-3-entrypoint-riêng-biệt)
- [Mở rộng project](#mở-rộng-project)
- [Ghi chú kỹ thuật theo từng phương pháp](#ghi-chú-kỹ-thuật-theo-từng-phương-pháp)
- [License / nguồn gốc code](#license--nguồn-gốc-code)
- [Trạng thái kiểm chứng](#trạng-thái-kiểm-chứng)

## Cấu trúc project

```
TTA-Proposal-Benchmark/
├── configs/
│   ├── methods/                  # sieu tham so tung phuong phap (tent/eata/sar/deyo/memo/bitta_*.yaml)
│   ├── datasets/                 # thiet lap tung dataset (5 file .yaml)
│   └── settings/                 # 3 setting TTA: fully_tta / continuous / mixed_shift
├── src/
│   ├── main.py                   # entrypoint chung cho ho BiTTA-family (TENT/EATA/SAR/BiTTA)
│   ├── conf.py                   # cau hinh dataset (opt dict), khong sua so voi ban goc
│   ├── process_cifar.py          # tien xu ly CIFAR-10-C/100-C tu file .npy Zenodo
│   ├── learner/                  # TENT, EATA, SAR, CoTTA, RoTTA, SoTTA, SimATTA, BiTTA (dnn.py + bitta.py)
│   ├── models/                   # ResNet.py (ResNet-18 co Dropout de MC-Dropout), ViT.py
│   ├── data_loader/               # loader cho CIFAR10/100, TinyImageNet, PACS, VLCS, ImageNet-R...
│   ├── utils/                     # loss functions, active_memory (Platt scaling C1), calibration...
│   └── methods/
│       ├── deyo/                  # ban sao nguyen ven repo chinh thuc cua tac gia DeYO
│       │   ├── main.py            # entrypoint rieng cho DeYO (kien truc adapt() khac BiTTA-family)
│       │   ├── config.py
│       │   ├── methods/           # deyo.py, tent.py, eata.py, sar.py, sam.py
│       │   ├── dataset/           # waterbirds_dataset.py, ColoredMNIST_dataset.py, *.npy (label shifts)
│       │   ├── models/            # Res.py, resnet.py
│       │   └── utils/             # utils.py, cli_utils.py, third_party.py
│       └── memo/                  # MEMO -- vong lap episodic rieng, tu chua
│           ├── main.py            # entrypoint rieng, dung lai checkpoint ResNet-18 cua BiTTA-family
│           └── utils/             # augmix.py (augmentation loi cua thuat toan), data.py (doc dataset)
├── scripts/
│   ├── smoke_test.sh              # kiem tra nhanh 1 lenh voi --nsample 500 truoc khi chay full
│   ├── run_bitta_family.sh        # chay 1 method (tent|eata|sar|bitta_baseline|bitta_proposal) x 1 dataset x 1 seed
│   ├── run_deyo_family.sh         # chay DeYO/TENT/EATA/SAR theo runner rieng cua DeYO
│   ├── run_memo_family.sh         # chay MEMO (hoac baseline no_adapt) theo runner rieng cua MEMO
│   └── run_all_bitta_family.sh    # vong lap toan bo method x dataset x seed
├── dataset/                       # (rong) CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C, WaterBirds
├── domainbed_dataset/              # (rong) PACS -- ten thu muc khop dung conf.py goc, khong sua code
├── pretrained_weights/             # (rong) checkpoint theo tung dataset
├── log/                             # output ket qua (online_eval.json moi corruption/domain)
├── results/
│   ├── aggregate_results.py        # tong hop ho BiTTA-family (online_eval.json) thanh 1 bang
│   └── aggregate_deyo_results.py   # tong hop DeYO + MEMO (log van ban) thanh 1 bang
├── requirements.txt
└── .gitignore
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.9, PyTorch ≥ 2.0 với build CUDA phù hợp GPU cá nhân (dự án được thiết
kế để chạy trên máy có 1 GPU ≥ 8 GB VRAM, không phụ thuộc cluster/multi-GPU).

## Chuẩn bị dữ liệu

Thư mục `dataset/`, `domainbed_dataset/`, `pretrained_weights/` hiện để trống theo đúng
yêu cầu ban đầu của đề án (chưa có nguồn dữ liệu). Mỗi thư mục con đều có `README.md`
hướng dẫn nguồn tải và cấu trúc thư mục cần có cho: CIFAR-10-C, CIFAR-100-C,
Tiny-ImageNet-C, PACS, WaterBirds.

## Cách chạy

### 1. Kiểm tra nhanh (bắt buộc trước khi chạy full)

```bash
bash scripts/smoke_test.sh
```

### 2. Họ BiTTA — TENT / EATA / SAR / BiTTA-baseline / BiTTA-Proposal

```bash
bash scripts/run_bitta_family.sh bitta_proposal cifar10_c 0 final_10_test_0_dist1
bash scripts/run_bitta_family.sh tent            cifar10_c 0 tent_cifar10_s0
```

### 3. DeYO (hoặc TENT/EATA/SAR theo runner gốc của DeYO)

```bash
bash scripts/run_deyo_family.sh deyo ImageNet-C normal 2024
bash scripts/run_deyo_family.sh deyo Waterbirds spurious 2024
```

### 4. MEMO — episodic, dùng lại checkpoint ResNet-18 của họ BiTTA-family

```bash
bash scripts/run_memo_family.sh memo cifar10 gaussian_noise 0 5
bash scripts/run_memo_family.sh memo pacs sketch 0
bash scripts/run_memo_family.sh no_adapt cifar10 gaussian_noise 0 5   # baseline khong adapt
```

### 5. Chạy toàn bộ

Rất nặng — nên chạy từng dòng để kiểm tra khi test lần đầu.

```bash
bash scripts/run_all_bitta_family.sh
```

### 6. Tổng hợp bảng so sánh

```bash
cd results
python aggregate_results.py --dataset cifar10_c                                       # TENT/EATA/SAR/BiTTA-baseline/BiTTA-Proposal
python aggregate_deyo_results.py --log_dir ../log/deyo_deyo_ImageNet-C_normal_2024     # DeYO
python aggregate_deyo_results.py --log_dir ../log/memo_memo_cifar10_s0                 # MEMO
```

`aggregate_deyo_results.py` dùng chung cho DeYO và MEMO vì cả hai ghi log dạng văn bản
(dòng `"Result under ... average: X.XXXXX"`), khác cấu trúc `online_eval.json` của họ
BiTTA-family — xem [Định dạng log](#ghi-chú-kỹ-thuật-theo-từng-phương-pháp) bên dưới.

Kết quả in ra dạng:

```
Method                     Mean Acc (%)       Std   # seed hoan tat
-----------------------------------------------------------------------
TENT                              80.49      0.xx              3/3
EATA                              75.17      0.xx              3/3
SAR                               83.78      0.xx              3/3
BiTTA (baseline)                  87.23      0.25              3/3
BiTTA-Proposal (C1+C4)            87.29      0.28              3/3
```

## 3 setting TTA (mở rộng được)

| Setting | BiTTA-family | DeYO-family | MEMO |
|---|---|---|---|
| Fully TTA | Mặc định (mọi method) | `--continual False` (mặc định) | Mặc định — duy nhất khả dụng |
| Continuous | `--tgt cont` | `--continual True --exp_type normal` | Không áp dụng |
| Mixed shift | `--tgt cont --random_setting` | `--continual True --exp_type mix_shifts` | Không áp dụng |

MEMO chỉ hỗ trợ Fully TTA vì đây là đặc tính thuật toán, không phải giới hạn tích hợp:
MEMO nạp lại đúng checkpoint gốc trước mỗi ảnh test (episodic, Algorithm 1 của paper) —
không có khái niệm "trạng thái mang sang batch/domain kế tiếp" để mà reset hay không
reset. Áp đặt một biến thể "continuous" cho MEMO sẽ đi ngược lại đúng thiết kế đã công
bố của thuật toán.

Chi tiết và trạng thái (đã chạy / TODO) của từng setting nằm trong `configs/settings/*.yaml`.

## Kiến trúc: vì sao 3 entrypoint riêng biệt

`src/main.py` (họ BiTTA), `src/methods/deyo/main.py` (DeYO) và `src/methods/memo/main.py`
(MEMO) có vòng lặp `adapt()` khác nhau về bản chất:

- **BiTTA-family**: bộ nhớ FIFO hai chiều (`M_C`/`M_I`) + policy gradient dual-path, liên
  tục (continual) qua nhiều batch.
- **DeYO**: lọc mẫu theo entropy + PLPD trên từng batch độc lập, không có bộ nhớ xuyên batch.
- **MEMO**: episodic hoàn toàn — mỗi ảnh test được nạp lại đúng checkpoint gốc, augment
  thành một batch augmix rồi tối thiểu hoá marginal entropy chỉ cho riêng ảnh đó, không
  mang trạng thái sang ảnh kế tiếp.

Gộp chung một vòng lặp sẽ phải viết lại thuật toán của các phía, rủi ro sai lệch với paper
gốc. Do đó dự án giữ nguyên ba runner đã được xác thực riêng, và dùng
`results/aggregate_results.py` + `results/aggregate_deyo_results.py` làm lớp tổng hợp
chung ở output.

## Mở rộng project

### Thêm dataset mới cho họ BiTTA-family

1. Thêm entry `opt` mới vào `src/conf.py` (theo mẫu `CIFAR10Opt`/`PACSOpt` đã có).
2. Viết class Dataset mới trong `src/data_loader/` (xem `PACSDataset.py` làm mẫu cho
   dataset dạng `ImageFolder`, hoặc `CIFAR10Dataset.py` cho dạng `.npy`).
3. Đăng ký nhánh `elif '<ten_dataset>' in conf.args.dataset:` trong `src/main.py` (khu
   vực chọn `opt`).
4. Nếu dùng chế độ `--tgt cont`, thêm `CONT_SEQUENCE_<TEN>` vào `conf.py` và đăng ký
   nhánh tương ứng trong `main.py`.

Ví dụ cụ thể cho WaterBirds: đã có sẵn loader ở
`src/methods/deyo/dataset/waterbirds_dataset.py`, chỉ cần viết adapter theo 4 bước trên.

### Thêm phương pháp TTA mới

Đặt class mới kế thừa `DNN` (`src/learner/dnn.py`) vào `src/learner/<ten>.py` (xem
`tent.py` — ngắn nhất — làm mẫu), rồi đăng ký `elif conf.args.method == "<TEN>":` trong
`src/main.py`. Override `test_time_adaptation()` để cài thuật toán riêng.

### Model

Backbone chuẩn cho toàn bộ dự án là **ResNet-18** (`src/models/ResNet.py`, class
`ResNetDropout18`) — có sẵn lớp Dropout sau mỗi residual block để phục vụ MC-Dropout
(dùng bởi BiTTA và làm nền tảng ước lượng độ tin cậy cho các phương pháp khác). Không
cần đổi kiến trúc khi thêm dataset hoặc method mới.

## Ghi chú kỹ thuật theo từng phương pháp

### DeYO — các sửa đổi wiring so với repo gốc

`src/methods/deyo/main.py` có 2 chỗ được sửa so với repo gốc của tác giả. Cả hai đều là
lựa chọn kiến trúc mạng (model selection); không đụng đến `methods/deyo.py`,
`methods/tent.py`, `methods/eata.py`, `methods/sar.py` (các file cài đặt thuật toán, giữ
nguyên 100%):

1. **Nhánh `resnet18_bn`** cho `net` (dòng ~361) và `net_ewc` (dòng ~280): bản gốc chỉ xử
   lý đúng trường hợp `dset == 'ColoredMNIST'` (tải một file pickle có sẵn) — nếu chạy
   `resnet18_bn` với dataset khác, biến `net`/`net_ewc` sẽ không được gán, gây lỗi
   `UnboundLocalError` khi thực thi `net.cuda()`. Đã bổ sung nhánh tổng quát
   `Resnet.__dict__['resnet18'](pretrained=True)` cho các dataset còn lại — cùng cách gọi
   mà bản gốc đã dùng cho `resnet50_bn_torch`.
2. **Assert tại `if args.dset == 'Waterbirds':`** (dòng ~190): bản gốc ép cứng
   `args.model == 'resnet50_bn_torch'`. Đã nới thành
   `in ('resnet50_bn_torch', 'resnet18_bn')`.

> **Cảnh báo khoa học:** tổ hợp DeYO + ResNet-18 + Waterbirds (và các dataset kiểu
> ImageNet-C) là phần mở rộng để nhất quán với backbone ResNet-18 xuyên suốt đề án,
> **không phải** cấu hình đã được paper DeYO kiểm chứng (paper gốc chỉ báo cáo
> ResNet-50-BN cho Waterbirds — Table 3). Khi báo cáo kết quả từ tổ hợp này trong luận
> văn, cần nêu rõ đây là thực nghiệm mở rộng, không phải tái lập số liệu paper.

### MEMO — dùng lại checkpoint ResNet-18 của họ BiTTA-family

Khác với DeYO (tự quản lý checkpoint riêng), `src/methods/memo/main.py` nạp trực tiếp
`pretrained_weights/<dataset>/cp_last_<seed>.pth.tar` — cùng file mà TENT/EATA/SAR/BiTTA
dùng. Điều này khả thi vì `ResNetDropout18` (`src/models/ResNet.py`) chỉ thêm `forward()`
tuỳ biến, không đổi tên tham số so với `torchvision.models.resnet18` gốc, nên state_dict
tương thích 1:1.

`build_model()` trong `main.py` của MEMO lặp lại đúng trình tự khởi tạo của
`src/learner/dnn.py` (thay `fc` theo `num_class` trước khi `load_state_dict`) và áp dụng
đúng hệ số chuẩn hoá (mean/std) mà checkpoint đó được train cùng — lấy từ
`src/utils/normalize_layer.py` (CIFAR-10: `[0.4914, 0.4822, 0.4465]` /
`[0.2471, 0.2435, 0.2616]`, CIFAR-100 và các dataset độ phân giải ImageNet tương tự).
Các hệ số này được đọc và sao chép trực tiếp — không import chéo — để giữ `memo/` tự chứa
độc lập như `deyo/`.

Nhờ vậy cả 7 phương pháp trong đề án xuất phát từ đúng một bộ trọng số pretrained, đảm
bảo so sánh công bằng.

### Định dạng log khác nhau giữa các họ phương pháp

- **Họ BiTTA** (`src/main.py`):
  `log/<dataset>/BiTTA/tgt_cont/<log_prefix>/<unit>/online_eval.json` — JSON có cấu trúc,
  đọc bằng `results/aggregate_results.py`.
- **DeYO** (`src/methods/deyo/main.py`) và **MEMO** (`src/methods/memo/main.py`): file
  `.txt` ghi dòng kết quả
  `"Result under <corruption>-<level>. The adaptation accuracy of <METHOD> is  average: X.XXXXX"`
  trong thư mục `--output` đã chỉ định — cả hai đọc chung bằng
  `results/aggregate_deyo_results.py` (dùng regex, không phân biệt method).

## License / nguồn gốc code

| Thành phần | Nguồn gốc |
|---|---|
| `src/main.py`, `src/conf.py`, `src/learner/*`, `src/data_loader/*`, `src/models/*`, `src/utils/*` | Repo BiTTA gốc của đề án — không chỉnh sửa logic thuật toán |
| `src/methods/deyo/*` | Sao chép nguyên vẹn từ repo chính thức của tác giả DeYO (xem `LICENSE` và `DEYO_ORIGINAL_README.md` trong cùng thư mục) |
| `src/methods/memo/*` | Viết lại tự chứa (self-contained) dựa trên đúng thuật toán và augmentation pipeline (augmix) của repo chính thức MEMO (Zhang et al., NeurIPS 2022, [github.com/zhangmarvin/memo](https://github.com/zhangmarvin/memo)) — hàm `marginal_entropy` / `adapt_single` / augmix trong `main.py` và `utils/augmix.py` sao chép logic gốc; phần đọc dữ liệu (`utils/data.py`) và việc dùng lại checkpoint ResNet-18 của BiTTA-family là phần viết mới để khớp với cấu trúc dataset/checkpoint sẵn có trong project |

## Trạng thái kiểm chứng

- Chuỗi import (`conf`, `learner.*`, `data_loader.*`, `models.*`, `utils.*`) đã chạy test
  thực tế, xác nhận đúng.
- `src/main.py` chạy được tới bước parse argument + chọn device (dừng ở
  `torch.cuda.set_device` do máy build project này không có PyTorch bản CUDA — sẽ chạy
  bình thường trên máy có GPU thật).
- `src/methods/memo/main.py`: đã test thực tế `build_model()` (nạp state_dict giả lập
  vào `torchvision.models.resnet18` sau khi thay `fc`) và `marginal_entropy()` trên CPU,
  cùng pipeline augmix (`utils/augmix.py`) trên ảnh giả lập — cả hai chạy đúng, không lỗi.
- Chưa có dữ liệu / checkpoint thật (theo đúng yêu cầu ban đầu) — xem mục
  [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu) trước khi chạy full.
