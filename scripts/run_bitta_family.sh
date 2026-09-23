#!/bin/bash
# Chay 1 (method, dataset, setting, seed) cua kich ban thuc nghiem.
#
# Cach dung: bash run_bitta_family.sh <method> <dataset> <setting> <seed> [log_prefix]
#   method : tent | eata | sar | bitta_baseline | bitta_proposal
#            (tent/eata/sar LUON chay o che do co binary feedback: --enable_bitta, "TENT*/EATA*/SAR*" cua paper)
#   dataset: cifar10_c | cifar100_c | tiny_imagenet_c | pacs | imagenet_r | colored_mnist | waterbirds
#   setting: continuous | mixed | fully
#
# Ma tran hop le (script tu choi to hop khac):
#   continuous : cifar10_c cifar100_c tiny_imagenet_c pacs
#   mixed      : cifar10_c pacs                       (data_loader chi ho tro danh sach nhieu domain cho 2 dataset nay)
#   fully      : cifar10_c cifar100_c tiny_imagenet_c pacs imagenet_r colored_mnist waterbirds
#
# Setting duoc cai dat nhu sau:
#   continuous: 1 tien trinh, --tgt cont (15 corruption / 3 domain PACS noi tiep, KHONG reset mo hinh)
#   mixed     : 1 tien trinh, --tgt cont --random_setting (cac corruption/domain tron ngau nhien trong 1 luong)
#   fully     : moi corruption/domain la 1 tien trinh RIENG, khoi dong tu mo hinh nguon (--tgt <unit>);
#               imagenet_r (--tgt corrupt), colored_mnist (--tgt test) va waterbirds (--tgt test)
#               chi co 1 domain dich.
#
# Bien moi truong tuy chon:
#   NSAMPLE=<n>      gioi han so anh dich moi corruption/domain (imagenet_r mac dinh 10000: main.py giu TOAN BO tensor
#                    float32 3x224x224 ~ 0.6 MB/anh trong RAM; 30.000 anh ~ 18 GB, dinh diem gap doi khi stack)
#   FISHER_SIZE=<n>  so anh tinh Fisher cua EATA (mac dinh 2000; ha xuong khi smoke test tren tap nho)
#
# Sieu tham so theo script goc tta.sh cua repo BiTTA. RIENG imagenet_r va colored_mnist KHONG co cau hinh nao trong
# paper/tta.sh cho BiTTA (va colored_mnist cho EATA): dung cau hinh cua dataset tuong dong (xem ghi chu) -- CHUA tinh chinh.
#   - tent/eata/sar: memory_type FIFO (BAT BUOC: learner/tent.py, sar.py, eata.py assert mem la FIFO) + --enable_bitta.
#   - bitta_*: memory_type ActivePriorityFIFO + MC-Dropout + --use_learned_stats.
# Vi du: bash run_bitta_family.sh bitta_proposal cifar10_c continuous 0
#        bash run_bitta_family.sh tent pacs mixed 0
#        bash run_bitta_family.sh eata imagenet_r fully 1

set -e

METHOD=$1
DATASET=$2
SETTING=$3
SEED=$4
LOG_PREFIX=${5:-${METHOD}_${DATASET}_${SETTING}_s${SEED}}

if [ $# -lt 4 ]; then
  sed -n 2,4p "$0"; exit 1
fi

cd "$(dirname "$0")/.."   # ve thu muc goc project

CORR15=(gaussian_noise-5 shot_noise-5 impulse_noise-5 defocus_blur-5 glass_blur-5 motion_blur-5 zoom_blur-5
        snow-5 frost-5 fog-5 brightness-5 contrast-5 elastic_transform-5 pixelate-5 jpeg_compression-5)
CKPT_ARG=(); NS_ARG=()
case $DATASET in
  cifar10_c)
    DS=cifar10; MODEL=resnet18; UNITS=("${CORR15[@]}"); MIXED_OK=1
    EPOCH=3; LR=0.0001; DROPOUT=0.3; NDROP=4; RST=0.0
    EATA_LR=0.005; EATA_EM=0.92103; EATA_DM=0.4; EATA_FA=1
    CKPT_ARG=(--load_checkpoint_path pretrained_weights/cifar10/cp_last_${SEED}.pth.tar) ;;
  cifar100_c)
    DS=cifar100; MODEL=resnet18; UNITS=("${CORR15[@]}"); MIXED_OK=0
    EPOCH=3; LR=0.0001; DROPOUT=0.3; NDROP=4; RST=0.0
    EATA_LR=0.005; EATA_EM=1.84207; EATA_DM=0.4; EATA_FA=1
    CKPT_ARG=(--load_checkpoint_path pretrained_weights/cifar100/cp_last_${SEED}.pth.tar) ;;
  tiny_imagenet_c)
    DS=tiny-imagenet; MODEL=resnet18_pretrained; UNITS=("${CORR15[@]}"); MIXED_OK=0
    EPOCH=5; LR=0.00005; DROPOUT=0.1; NDROP=2; RST=0.01
    EATA_LR=0.001; EATA_EM=2.1193; EATA_DM=0.5; EATA_FA=2000
    CKPT_ARG=(--load_checkpoint_path pretrained_weights/tiny-imagenet/cp_last_${SEED}.pth.tar) ;;
  pacs)
    DS=pacs; MODEL=resnet18_pretrained; UNITS=(art_painting cartoon sketch); MIXED_OK=1
    EPOCH=3; LR=0.001; DROPOUT=0.3; NDROP=2; RST=0.0
    EATA_LR=0.001; EATA_EM=0.7784; EATA_DM=0.5; EATA_FA=2000
    CKPT_ARG=(--load_checkpoint_path pretrained_weights/pacs/cp_last.pth.tar) ;;
  imagenet_r)
    # 200 lop, ResNet-18 pretrained torchvision loc 1000->200 logit (khong can checkpoint). Cau hinh BiTTA muon cua
    # Tiny-ImageNet-C (cung 224px, cung backbone, ~200 lop); EATA: e_margin = 0.4*ln(200), cac tham so con lai theo
    # bo ImageNet cua tta.sh.
    DS=imagenetR; MODEL=resnet18_pretrained; UNITS=(corrupt); MIXED_OK=0
    EPOCH=5; LR=0.00005; DROPOUT=0.1; NDROP=2; RST=0.01
    EATA_LR=0.00025; EATA_EM=2.1193; EATA_DM=0.05; EATA_FA=2000
    NSAMPLE=${NSAMPLE:-10000} ;;
  colored_mnist)
    # 2 lop, anh 28x28. Cau hinh BiTTA/EATA muon cua CIFAR (anh nho), e_margin = 0.4*ln(2). Checkpoint: huan luyen
    # nguon bang --method Src --src all_train (xem pretrained_weights/README.md).
    DS=colored-mnist; MODEL=resnet18_pretrained; UNITS=(test); MIXED_OK=0
    EPOCH=3; LR=0.0001; DROPOUT=0.3; NDROP=4; RST=0.0
    EATA_LR=0.005; EATA_EM=0.27726; EATA_DM=0.4; EATA_FA=1
    CKPT_ARG=(--load_checkpoint_path pretrained_weights/colored-mnist/cp_last_${SEED}.pth.tar) ;;
  waterbirds)
    # 2 lop (landbird/waterbird), anh 224x224, tuong quan gia nen-nhan. Cau hinh BiTTA/EATA muon cua
    # Tiny-ImageNet-C (cung 224px, cung backbone); EATA: e_margin = 0.4*ln(2). Checkpoint: huan luyen nguon
    # bang --method Src --src train (xem pretrained_weights/README.md). CHI Fully TTA (1 domain dich: test).
    DS=waterbirds; MODEL=resnet18_pretrained; UNITS=(test); MIXED_OK=0
    EPOCH=5; LR=0.00005; DROPOUT=0.1; NDROP=2; RST=0.01
    EATA_LR=0.005; EATA_EM=0.27726; EATA_DM=0.4; EATA_FA=1
    CKPT_ARG=(--load_checkpoint_path pretrained_weights/waterbirds/cp_last.pth.tar) ;;
  *)
    echo "Dataset khong hop le: $DATASET"; exit 1 ;;
esac

case $SETTING in
  continuous)
    if [ ${#UNITS[@]} -lt 2 ]; then echo "'$DATASET' chi co 1 domain dich -> khong co setting continuous"; exit 1; fi ;;
  mixed)
    if [ "$MIXED_OK" != "1" ]; then echo "Mixed shift chi ho tro cifar10_c va pacs (khong ho tro $DATASET)"; exit 1; fi ;;
  fully) ;;
  *)
    echo "Setting khong hop le: $SETTING (continuous | mixed | fully)"; exit 1 ;;
esac

if [ -n "${NSAMPLE:-}" ]; then NS_ARG=(--nsample "$NSAMPLE"); fi

# Tham so rieng cua tung method (KHONG dung chung: xem ghi chu dau file)
case $METHOD in
  tent)
    EXTRA=(--method TENT --epoch 1 --lr 0.001 --memory_type FIFO --enable_bitta) ;;
  eata)
    EXTRA=(--method EATA --epoch 1 --lr "$EATA_LR" --memory_type FIFO --enable_bitta
           --e_margin "$EATA_EM" --d_margin "$EATA_DM" --fisher_size "${FISHER_SIZE:-2000}" --fisher_alpha "$EATA_FA") ;;
  sar)
    EXTRA=(--method SAR --epoch 1 --lr 0.00025 --memory_type FIFO --enable_bitta) ;;
  bitta_baseline)
    EXTRA=(--method BiTTA --epoch "$EPOCH" --lr "$LR" --memory_type ActivePriorityFIFO
           --restoration_factor "$RST" --use_learned_stats --bn_momentum 0.3
           --dropout_rate "$DROPOUT" --n_dropouts "$NDROP" --sample_selection mc_conf) ;;
  bitta_proposal)
    EXTRA=(--method BiTTA --epoch "$EPOCH" --lr "$LR" --memory_type ActivePriorityFIFO
           --restoration_factor "$RST" --use_learned_stats --bn_momentum 0.3
           --dropout_rate "$DROPOUT" --n_dropouts "$NDROP"
           --sample_selection info_max --n_dropouts_select 16 --explore_eps 0.0 --platt_min_samples 20) ;;
  *)
    echo "Method khong hop le: $METHOD"; exit 1 ;;
esac

run_main() {  # $1 = gia tri --tgt, $2... = co them
  local TGT=$1; shift
  python src/main.py --gpu_idx 0 --dataset "$DS" --tgt "$TGT" \
      --model "$MODEL" "${CKPT_ARG[@]}" \
      --seed "$SEED" --remove_cp --online --tgt_train_dist 1 \
      --update_every_x 64 --memory_size 64 --weight_decay 0 \
      "${EXTRA[@]}" "${NS_ARG[@]}" "$@" \
      --log_name log/ --log_prefix "$LOG_PREFIX"
}

case $SETTING in
  continuous) run_main cont ;;
  mixed)      run_main cont --random_setting ;;
  fully)
    for U in "${UNITS[@]}"; do
      echo ">>> fully TTA | $DATASET | $U"
      run_main "$U"
    done ;;
esac
