#!/bin/bash
# Cach dung: bash run_bitta_family.sh <method> <dataset> <seed> <log_prefix>
#   method:  tent | eata | sar | bitta_baseline | bitta_proposal
#   dataset: cifar10_c | cifar100_c | tiny_imagenet_c | pacs
# Vi du:     bash run_bitta_family.sh bitta_proposal cifar10_c 0 final_10_test_0_dist1

set -e

METHOD=$1        # tent | eata | sar | bitta_baseline | bitta_proposal
DATASET=$2       # cifar10_c | cifar100_c | tiny_imagenet_c | pacs
SEED=$3
LOG_PREFIX=$4

cd "$(dirname "$0")/.."   # ve thu muc goc project

case $DATASET in
  cifar10_c)
    DS=cifar10; MODEL=resnet18; EPOCH=3; LR=0.0001; DROPOUT=0.3; NDROP=4; RST=0.0; DIST=1
    CKPT=pretrained_weights/cifar10/cp_last_${SEED}.pth.tar ;;
  cifar100_c)
    DS=cifar100; MODEL=resnet18; EPOCH=3; LR=0.0001; DROPOUT=0.3; NDROP=4; RST=0.0; DIST=1
    CKPT=pretrained_weights/cifar100/cp_last_${SEED}.pth.tar ;;
  tiny_imagenet_c)
    DS=tiny-imagenet; MODEL=resnet18_pretrained; EPOCH=5; LR=0.00005; DROPOUT=0.1; NDROP=2; RST=0.01; DIST=1
    CKPT=pretrained_weights/tiny-imagenet/cp_last_${SEED}.pth.tar ;;
  pacs)
    DS=pacs; MODEL=resnet18_pretrained; EPOCH=3; LR=0.001; DROPOUT=0.3; NDROP=2; RST=0.0; DIST=1
    CKPT=pretrained_weights/pacs/cp_last.pth.tar ;;
  *)
    echo "Dataset khong hop le: $DATASET"; exit 1 ;;
esac

case $METHOD in
  tent)
    EXTRA="--method TENT" ;;
  eata)
    EXTRA="--method EATA --fisher_size 2000 --fisher_alpha 2000.0" ;;
  sar)
    EXTRA="--method SAR --lr 0.00025" ;;
  bitta_baseline)
    EXTRA="--method BiTTA --sample_selection mc_conf" ;;
  bitta_proposal)
    EXTRA="--method BiTTA --sample_selection info_max --n_dropouts_select 16 --explore_eps 0.0 --platt_min_samples 20" ;;
  *)
    echo "Method khong hop le: $METHOD"; exit 1 ;;
esac

python src/main.py --gpu_idx 0 --dataset "$DS" --tgt cont \
    --model "$MODEL" --load_checkpoint_path "$CKPT" \
    --epoch "$EPOCH" --seed "$SEED" --remove_cp --online --tgt_train_dist "$DIST" \
    --update_every_x 64 --memory_size 64 --memory_type ActivePriorityFIFO \
    --weight_decay 0 --lr "$LR" --restoration_factor "$RST" \
    --use_learned_stats --bn_momentum 0.3 --dropout_rate "$DROPOUT" \
    --n_dropouts "$NDROP" \
    $EXTRA \
    --log_name log/ --log_prefix "$LOG_PREFIX"
