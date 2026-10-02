# Kiểm chứng Tầng 1 + Tầng 2: tiền đề chọn mẫu của BiTTA gốc

Kiểm tra bằng số liệu thật claim: *"những mẫu có confidence thấp thường sai, confidence cao thường
đúng, nên cách chọn mẫu luôn-chọn-confidence-thấp-nhất của BiTTA gốc thường không hiệu quả bằng
Proposal (chọn theo entropy nhị phân lớn nhất)"*.

## Phương pháp (áp dụng giống nhau cho cả 4 dataset)

1. Lấy ảnh **sạch thật** (không qua corruption benchmark chính thức) của mỗi dataset.
2. Tự áp **nhiễu Gaussian iid** cường độ `sigma` (mặc định 0,26 trên thang [0,1]) lên toàn bộ ảnh —
   **không phải** file `.npy`/ảnh corrupted chính thức của CIFAR-10-C/CIFAR-100-C/Tiny-ImageNet-C
   (các bộ gốc trên Zenodo/Kaggle không tách riêng được từng loại nhiễu để tải nhanh cho một kiểm
   chứng sơ bộ — nếu cần số liệu chính thức để đưa vào báo cáo, nên đối chiếu lại bằng dữ liệu
   corrupted thật của từng benchmark).
3. Nạp đúng checkpoint và kiến trúc mà dự án đã huấn luyện (`pretrained_weights/<dataset>/...`).
4. Xử lý ảnh theo **từng batch `update_every_x`=64** theo đúng thứ tự, và trước khi tính confidence,
   **cập nhật thống kê BatchNorm bằng batch đó** (`--use_learned_stats --bn_momentum 0.3`, đúng cấu
   hình thật của BiTTA/BiTTA-Proposal) — mô phỏng đúng trạng thái mô hình tại thời điểm
   `active_sample_selection()` thực sự hoạt động trong pipeline thật. **Không** mô phỏng bước cập
   nhật trọng số bằng gradient (BFA/ABA) từ vài mẫu phản hồi mỗi batch — đây là hiệu ứng bậc hai so
   với cập nhật BatchNorm, và cố ý bỏ qua để không gắn kết quả vào một thuật toán chọn mẫu cụ thể.
5. Tính `C(x)` = trung bình softmax qua `n_dropouts` lượt MC-Dropout tại lớp dự đoán — đúng công
   thức BiTTA gốc dùng để chọn mẫu (`dropout_rate`/`n_dropouts` lấy đúng từ
   `scripts/run_bitta_family.sh` cho từng dataset).
6. Gộp `(C(x), đúng/sai)` của toàn bộ ảnh, qua 3 seed, chia 10 khoảng confidence, tính:
   - **Tầng 1**: accuracy thực nghiệm trong từng khoảng (đường cong hiệu chuẩn).
   - **Tầng 2**: entropy nhị phân `h_b` của accuracy đó trong từng khoảng (thông tin kỳ vọng nếu hỏi
     oracle một mẫu trong khoảng đó).

## Cấu trúc thư mục

```
verify/
  README.md            # file nay
  common.py            # ham dung chung: cap nhat BN, tinh confidence, chia bin/entropy, ve hinh
  datasets.py           # cau hinh rieng 4 dataset (checkpoint, dropout, kich thuoc anh, nap anh sach)
  premise_check.py      # script chinh, chay qua dong lenh
  .cache/               # (tu sinh) du lieu CIFAR tai ve qua torchvision, khong dua len git
  results/              # (tu sinh) file .csv + .png ket qua, khong dua len git (tru .gitkeep)
```

## Cách chạy

Chạy từ thư mục gốc `TTA-Proposal-Benchmark/` (để đường dẫn checkpoint mặc định đúng).

### CIFAR-10-C và CIFAR-100-C (dễ nhất — tự tải qua torchvision, không cần chuẩn bị gì)

```bash
python verify/premise_check.py --dataset cifar10_c
python verify/premise_check.py --dataset cifar100_c
```

Lần chạy đầu sẽ tự tải CIFAR-10/CIFAR-100 (khoảng 170MB mỗi bộ) vào `verify/.cache/`. Toàn bộ 10.000
ảnh test × 3 seed, chạy trên CPU khoảng vài phút; nếu máy có GPU, script tự dùng (`--device cuda`
cũng được truyền thủ công nếu cần).

### Tiny-ImageNet-C (cần trỏ `--data-root` tới thư mục ảnh sạch 200 lớp)

Dùng đúng thư mục ảnh sạch (clean) mà checkpoint `pretrained_weights/tiny-imagenet/cp_last_<seed>.pth.tar`
đã được huấn luyện — mỗi lớp một thư mục con, dạng `ImageFolder` của torchvision.

**Nếu checkpoint không phải do pipeline này tự huấn luyện** (ví dụ lấy từ `pretrained_checkpoints`
của tác giả BiTTA gốc), script vẫn có thể dùng được, với điều kiện thư mục ảnh và checkpoint dùng
**cùng thứ tự gán chỉ số lớp**. Tiny-ImageNet-200 đặt tên thư mục lớp theo mã WordNet (`n01443537`,
...) và hầu hết pipeline (kể cả `ImageFolder`) gán chỉ số theo thứ tự chữ cái của tên thư mục — quy
ước gần như chuẩn chung của benchmark này, nên nhiều khả năng vẫn khớp. Để không phải đoán, script
**tự kiểm tra** bằng cách chạy checkpoint trên một ít ảnh sạch (mặc định 200 ảnh, `--sanity-check-n`)
**trước khi** thêm nhiễu và phân tích chính:
- Accuracy cao (gần mức mô hình nguồn thường đạt, ví dụ > 30–50%) → thứ tự lớp khớp, yên tâm chạy tiếp.
- Accuracy xấp xỉ mức ngẫu nhiên (`1/200 ≈ 0,5%`) → **thứ tự lớp lệch nhau**, script tự bỏ qua seed đó
  và in cảnh báo — khi đó cần tìm lại đúng thư mục ảnh (ví dụ bản `val/` chính thức của Tiny-ImageNet
  kèm `val_annotations.txt`, dựng lại theo đúng ánh xạ wnid → ảnh, thay vì đoán theo thứ tự thư mục).

```bash
python verify/premise_check.py --dataset tiny_imagenet_c \
    --data-root dataset/Tiny-ImageNet-C/origin/Data/train \
    --nsample 5000
```

Nếu bạn chạy trên máy/Kaggle chưa có sẵn thư mục này, cần chuẩn bị giống hệt bước Fully TTA trước đó
(symlink từ Kaggle dataset `akash2sharma/tiny-imagenet`, xem hướng dẫn Tiny-ImageNet-C trước đây),
hoặc trỏ `--data-root` tới bất kỳ thư mục `train` 200-lớp nào của Tiny-ImageNet gốc.

`--nsample 5000` giới hạn số ảnh sạch lấy ngẫu nhiên trước khi thêm nhiễu (toàn bộ thư mục có
100.000 ảnh — quá lâu nếu chạy hết trên CPU). Tăng số này nếu muốn chính xác hơn và có đủ thời gian/GPU.

### PACS (cần trỏ `--data-root` tới domain "photo" — domain nguồn đã dùng để huấn luyện checkpoint)

```bash
python verify/premise_check.py --dataset pacs --data-root domainbed_dataset/PACS/photo
```

**Lưu ý riêng cho PACS**: chỉ có **1 checkpoint dùng chung** (`pretrained_weights/pacs/cp_last.pth.tar`),
không có 3 checkpoint theo từng seed như CIFAR/Tiny. Script vẫn chạy "3 seed" để giữ cùng định dạng
kết quả, nhưng cả 3 lần đều dùng **chung một mô hình**, chỉ khác nhau ở **lần rút nhiễu Gaussian**
(mỗi seed một lượt nhiễu ngẫu nhiên khác nhau) — tức độ biến thiên giữa 3 "seed" ở đây phản ánh
nhiễu ngẫu nhiên của phép đo, không phải biến thiên do huấn luyện mô hình khác nhau như ở CIFAR/Tiny.
Domain "photo" của PACS chỉ có khoảng 1.670 ảnh, nên không cần `--nsample`.

## Tham số có thể chỉnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--sigma` | 0.26 | Độ lệch chuẩn nhiễu Gaussian trên thang [0,1], áp giống nhau cho cả 4 dataset |
| `--nsample` | toàn bộ (5000 riêng cho tiny_imagenet_c) | Giới hạn số ảnh sạch trước khi thêm nhiễu |
| `--update-every-x` | 64 | Kích thước batch cập nhật BatchNorm, giống `--update_every_x` |
| `--bn-momentum` | 0.3 | Giống `--bn_momentum` của BiTTA/BiTTA-Proposal |
| `--n-bins` | 10 | Số khoảng chia confidence |
| `--out-dir` | `verify/results` | Nơi lưu `.csv`/`.png` |
| `--device` | `cuda` nếu có, không thì `cpu` | |
| `--sanity-check-n` | 200 | Số ảnh sạch dùng kiểm tra nhanh trước mỗi seed (phát hiện lệch thứ tự lớp) |
| `--skip-sanity-check` | tắt | Bỏ qua bước kiểm tra nhanh ở trên |

## Đọc kết quả

Mỗi lần chạy in ra một bảng 10 dòng (một dòng/khoảng confidence) gồm số mẫu, confidence trung bình,
accuracy thực nghiệm, và `h_b`; kèm hai dòng tóm tắt: khoảng thấp nhất (nơi BiTTA gốc luôn chọn) và
khoảng có `h_b` cao nhất (nơi nhiều thông tin nhất). File `.csv` lưu đầy đủ bảng này; file `.png` vẽ
hai đồ thị (đường cong hiệu chuẩn Tầng 1, và entropy theo confidence Tầng 2).

**Tiền đề được xác nhận** nếu: (a) accuracy tăng theo confidence ở Tầng 1, và (b) khoảng thấp nhất
(BiTTA gốc chọn) có `h_b` thấp hơn rõ rệt so với khoảng có `h_b` cao nhất ở Tầng 2 — tức BiTTA gốc
đang hỏi đúng những mẫu ít thông tin nhất.
