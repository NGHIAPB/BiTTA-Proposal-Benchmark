#!/bin/bash
# Chay nhanh (--nsample nho) de kiem tra loi shape/import/duong dan truoc khi chay full.
# Kiem tra 3 setting tren CIFAR-10-C voi BiTTA-Proposal va TENT*.
# Cach dung: bash scripts/smoke_test.sh          (can dataset/CIFAR-10-C va pretrained_weights/cifar10/cp_last_0.pth.tar)
set -e
cd "$(dirname "$0")/.."

COMMON="--gpu_idx 0 --dataset cifar10 --model resnet18 --load_checkpoint_path pretrained_weights/cifar10/cp_last_0.pth.tar \
  --seed 0 --remove_cp --online --tgt_train_dist 1 --update_every_x 64 --memory_size 64 --weight_decay 0 --nsample 500 \
  --log_name log/"
BITTA="--method BiTTA --epoch 3 --lr 0.0001 --memory_type ActivePriorityFIFO --restoration_factor 0.0 --use_learned_stats \
  --bn_momentum 0.3 --dropout_rate 0.3 --n_dropouts 4 --sample_selection info_max --n_dropouts_select 16 \
  --explore_eps 0.0 --platt_min_samples 20"
TENT="--method TENT --epoch 1 --lr 0.001 --memory_type FIFO --enable_bitta"

python src/main.py $COMMON $BITTA --tgt cont --log_prefix smoke_continuous          # Continuous
python src/main.py $COMMON $BITTA --tgt cont --random_setting --log_prefix smoke_mixed   # Mixed shift
python src/main.py $COMMON $BITTA --tgt gaussian_noise-5 --log_prefix smoke_fully   # Fully TTA (1 corruption)
python src/main.py $COMMON $TENT  --tgt gaussian_noise-5 --log_prefix smoke_tent    # TENT* (co binary feedback)

echo "Smoke test OK -- xoa log rac:"
echo "  rm -rf log/cifar10/*/*/smoke_* log/cifar10/*/tgt_*/smoke_* cached_data"
