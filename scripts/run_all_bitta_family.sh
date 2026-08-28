#!/bin/bash
# Master script: chay het TENT/EATA/SAR/BiTTA-baseline/BiTTA-Proposal tren CIFAR-10-C,
# CIFAR-100-C, Tiny-ImageNet-C, PACS -- 3 seed (5 seed rieng cho PACS).
#
# CANH BAO: rat nang, nen chay tren may co GPU >=8GB VRAM va du thoi gian.
# Khuyen nghi chay tung dong 1 (comment bot) khi test lan dau.

set -e
cd "$(dirname "$0")"

METHODS=(tent eata sar bitta_baseline bitta_proposal)
DATASETS=(cifar10_c cifar100_c tiny_imagenet_c)
SEEDS=(0 1 2)

for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
      LOG_PREFIX="${m}_${ds}_s${s}"
      echo ">>> $m | $ds | seed=$s"
      bash run_bitta_family.sh "$m" "$ds" "$s" "$LOG_PREFIX"
    done
  done
done

# PACS: 5 seed (chuoi ngan, can nhieu seed hon de du power thong ke)
for m in "${METHODS[@]}"; do
  for s in 0 1 2 3 4; do
    LOG_PREFIX="${m}_pacs_s${s}"
    echo ">>> $m | pacs | seed=$s"
    bash run_bitta_family.sh "$m" pacs "$s" "$LOG_PREFIX"
  done
done

echo "Hoan tat. Chay tiep: python ../results/aggregate_results.py"
