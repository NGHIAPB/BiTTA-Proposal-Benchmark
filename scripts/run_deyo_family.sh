#!/bin/bash
# DeYO (Lee et al., ICLR 2024) -- chay 1 (dataset, setting, seed) cua kich ban thuc nghiem.
#
# Cach dung: bash run_deyo_family.sh <method> <dataset> <setting> <seed>
#   method : deyo (phuong phap chinh) | tent | eata | sar | no_adapt (cac phuong phap co san trong repo DeYO)
#   dataset: cifar10_c | cifar100_c | tiny_imagenet_c | pacs | imagenet_r | colored_mnist | waterbirds
#   setting: continuous | mixed | fully
#
# Ma tran hop le (script tu choi to hop khac):
#   continuous : cifar10_c cifar100_c tiny_imagenet_c pacs
#   mixed      : cifar10_c pacs
#   fully      : cifar10_c cifar100_c tiny_imagenet_c pacs imagenet_r colored_mnist waterbirds
#   (waterbirds CHI co o Fully TTA -- chi 1 domain dich "test" nen khong co chuoi domain de lien tuc/tron)
#
# Cai dat setting (xem src/methods/deyo/main.py):
#   continuous: 1 tien trinh, --continual True: mo hinh/optimizer KHONG reset giua cac corruption/domain.
#   mixed     : 1 tien trinh, --exp_type mix_shifts: cac domain noi thanh 1 luong (shuffle) roi thich nghi lien tuc.
#   fully     : moi corruption/domain la 1 luot rieng, moi luot bat dau lai tu checkpoint nguon (--continual False, --unit).
#
# Mo hinh/checkpoint/du lieu DUNG CHUNG voi ho BiTTA-family (--model resnet18_scenario, xem
# src/methods/deyo/dataset/scenario_dataset.py): thuat toan DeYO/TENT/EATA/SAR giu nguyen, chi noi lai data/model.
# Sieu tham so DeYO theo paper/script goc: deyo_margin 0.5, deyo_margin_e0 0.4, plpd_threshold 0.2, patch 4x4,
# batch 64, lr = 0.00025 (cong thuc goc voi bs>=32); RIENG waterbirds lr_mul=5 (giong DeYO goc cho Waterbirds).
# CHUA tinh chinh cho CIFAR/Tiny/PACS/ImageNet-R/ColoredMNIST (paper DeYO chi bao cao ImageNet-C/Waterbirds/ColoredMNIST).
#
# Vi du: bash run_deyo_family.sh deyo cifar10_c continuous 0
#        bash run_deyo_family.sh deyo pacs mixed 1
#        bash run_deyo_family.sh deyo waterbirds fully 0

set -e

METHOD=$1
DATASET=$2
SETTING=$3
SEED=$4

if [ -z "$METHOD" ] || [ -z "$DATASET" ] || [ -z "$SETTING" ] || [ -z "$SEED" ]; then
  echo "Cach dung: bash run_deyo_family.sh <method> <dataset> <setting> <seed>"; exit 1
fi
case $METHOD in deyo|tent|eata|sar|no_adapt) ;; *) echo "Method khong hop le: $METHOD"; exit 1 ;; esac

CORR15=(gaussian_noise-5 shot_noise-5 impulse_noise-5 defocus_blur-5 glass_blur-5 motion_blur-5 zoom_blur-5 snow-5 frost-5 fog-5 brightness-5 contrast-5 elastic_transform-5 pixelate-5 jpeg_compression-5)

LR_MUL=1; EXP_NORMAL=normal
case $DATASET in
  cifar10_c)       DSET=CIFAR10-C;   UNITS=("${CORR15[@]}"); MIXED_OK=1 ;;
  cifar100_c)      DSET=CIFAR100-C;  UNITS=("${CORR15[@]}"); MIXED_OK=0 ;;
  tiny_imagenet_c) DSET=TinyImageNet-C; UNITS=("${CORR15[@]}"); MIXED_OK=0 ;;
  pacs)            DSET=PACS-scenario; UNITS=(art_painting cartoon sketch); MIXED_OK=1 ;;
  imagenet_r)      DSET=ImageNetR-scenario; UNITS=(corrupt); MIXED_OK=0 ;;
  colored_mnist)   DSET=ColoredMNIST-scenario; UNITS=(test); MIXED_OK=0 ;;
  waterbirds)      DSET=WaterBirds-scenario; UNITS=(test); MIXED_OK=0; LR_MUL=5; EXP_NORMAL=spurious ;;
  *) echo "Dataset khong hop le: $DATASET"; exit 1 ;;
esac

case $SETTING in
  continuous)
    case $DATASET in cifar10_c|cifar100_c|tiny_imagenet_c|pacs) ;; *)
      echo "Continuous khong ap dung cho $DATASET (chi cifar10_c cifar100_c tiny_imagenet_c pacs)"; exit 1 ;; esac ;;
  mixed)
    [ "$MIXED_OK" = 1 ] || { echo "Mixed shift chi ap dung cho cifar10_c, pacs (khong phai $DATASET)"; exit 1; } ;;
  fully) ;;
  *) echo "Setting khong hop le: $SETTING"; exit 1 ;;
esac

cd "$(dirname "$0")/../src/methods/deyo"

OUT="../../../log/deyo_${METHOD}_${DATASET}_${SETTING}_s${SEED}"
COMMON=(--method "$METHOD" --dset "$DSET" --model resnet18_scenario --seed "$SEED"
        --test_batch_size 64 --workers 2 --wandb_log 0 --lr_mul "$LR_MUL"
        --aug_type patch --patch_len 4 --deyo_margin 0.5 --deyo_margin_e0 0.4 --plpd_threshold 0.2
        --output "$OUT")

case $SETTING in
  continuous)
    python main.py "${COMMON[@]}" --exp_type normal --continual True ;;
  mixed)
    python main.py "${COMMON[@]}" --exp_type mix_shifts --continual True ;;
  fully)
    for U in "${UNITS[@]}"; do
      python main.py "${COMMON[@]}" --exp_type "$EXP_NORMAL" --continual False --unit "$U"
    done ;;
esac
