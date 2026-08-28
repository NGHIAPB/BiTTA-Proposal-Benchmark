#!/bin/bash
# Cach dung: bash run_memo_family.sh <method> <dataset> <corruption> <seed> [level]
#   method:     no_adapt | memo
#   dataset:    cifar10 | cifar100 | tiny-imagenet | pacs
#   corruption: ten corruption (vd gaussian_noise) hoac ten domain PACS (vd sketch)
# Vi du:        bash run_memo_family.sh memo cifar10 gaussian_noise 0 5
#               bash run_memo_family.sh memo pacs sketch 0

set -e

METHOD=$1
DATASET=$2
CORRUPTION=$3
SEED=$4
LEVEL=${5:-5}

case $DATASET in
  cifar10)        CKPT=pretrained_weights/cifar10/cp_last_${SEED}.pth.tar; LR=0.005 ;;
  cifar100)       CKPT=pretrained_weights/cifar100/cp_last_${SEED}.pth.tar; LR=0.005 ;;
  tiny-imagenet)  CKPT=pretrained_weights/tiny-imagenet/cp_last_${SEED}.pth.tar; LR=0.00025 ;;
  pacs)           CKPT=pretrained_weights/pacs/cp_last.pth.tar; LR=0.00025 ;;
  *)
    echo "Dataset khong hop le: $DATASET"; exit 1 ;;
esac

cd "$(dirname "$0")/../src/methods/memo"

python main.py \
    --method "$METHOD" --dataset "$DATASET" --corruption "$CORRUPTION" --level "$LEVEL" \
    --seed "$SEED" --lr "$LR" \
    --checkpoint "../../../$CKPT" \
    --data_root ../../../dataset --domainbed_root ../../../domainbed_dataset \
    --output "../../../log/memo_${METHOD}_${DATASET}_s${SEED}"
