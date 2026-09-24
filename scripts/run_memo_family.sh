#!/bin/bash
# MEMO -- chi Fully TTA (thuat toan episodic: nap lai checkpoint goc truoc MOI anh test,
# khong mang trang thai sang anh/domain ke tiep -> Continuous va Mixed shift khong ap dung duoc).
#
# Cach dung: bash run_memo_family.sh <method> <dataset> <corruption> <seed> [level]
#   method:     no_adapt | memo
#   dataset:    cifar10 | cifar100 | tiny-imagenet | pacs | imagenet_r | colored_mnist | waterbirds
#   corruption: ten corruption (cifar/tiny-imagenet, vd gaussian_noise), ten domain PACS (vd sketch),
#               hoac ten split co dinh: "corrupt" (imagenet_r) | "test" (colored_mnist, waterbirds)
#   level:      muc do severity 1-5, chi dung cho cifar10/cifar100/tiny-imagenet (mac dinh 5)
#
# Bien moi truong tuy chon:
#   SHARDS=<k>    chay k tien trinh song song tren 1 GPU roi gop (ket qua tuong duong, nhanh ~k lan neu con CPU/GPU roi).
#   NGPU=<n>      so GPU de chia shard luan phien (mac dinh: tu nhan bang nvidia-smi -L).
#   PROFILE=<n>   do thoi gian n anh dau (augmix CPU / fwd-bwd GPU) roi thoat, khong ghi ket qua.
#   NSAMPLE=<n>   chi danh gia n anh NGAU NHIEN (theo seed); MEMO adapt tung anh (32 augmix/anh) nen
#                 dataset nhieu anh (tiny-imagenet, imagenet_r) rat lau -- nen dat vd NSAMPLE=2000-5000.
# Vi du:        bash run_memo_family.sh memo cifar10 gaussian_noise 0 5
#               bash run_memo_family.sh memo pacs sketch 0
#               NSAMPLE=5000 bash run_memo_family.sh memo tiny-imagenet gaussian_noise 0 5
#               bash run_memo_family.sh memo imagenet_r corrupt 0
#               bash run_memo_family.sh memo colored_mnist test 0
#               bash run_memo_family.sh memo waterbirds test 0

set -e

METHOD=$1
DATASET=$2
CORRUPTION=$3
SEED=$4
LEVEL=${5:-5}

# CKPT: duong dan tuong doi so voi thu muc goc project, hoac sentinel "imagenet_r" (= torchvision
# pretrained loc 1000->200 logit, khong can file)
case $DATASET in
  cifar10)        CKPT=pretrained_weights/cifar10/cp_last_${SEED}.pth.tar; LR=0.005 ;;
  cifar100)       CKPT=pretrained_weights/cifar100/cp_last_${SEED}.pth.tar; LR=0.005 ;;
  tiny-imagenet)  CKPT=pretrained_weights/tiny-imagenet/cp_last_${SEED}.pth.tar; LR=0.00025 ;;
  pacs)           CKPT=pretrained_weights/pacs/cp_last.pth.tar; LR=0.00025 ;;
  imagenet_r)     CKPT=imagenet_r; LR=0.00025 ;;   # khong can checkpoint
  colored_mnist)  CKPT=pretrained_weights/colored-mnist/cp_last_${SEED}.pth.tar; LR=0.005 ;;
  waterbirds)     CKPT=pretrained_weights/waterbirds/cp_last.pth.tar; LR=0.00025 ;;
  *)
    echo "Dataset khong hop le: $DATASET"; exit 1 ;;
esac

if [ "$CKPT" = "imagenet_r" ]; then CKPT_ARG="imagenet_r"; else CKPT_ARG="../../../$CKPT"; fi
NS_ARG=()
if [ -n "${NSAMPLE:-}" ]; then NS_ARG=(--nsample "$NSAMPLE"); fi

cd "$(dirname "$0")/../src/methods/memo"

ARGS=(--method "$METHOD" --dataset "$DATASET" --corruption "$CORRUPTION" --level "$LEVEL"
      --seed "$SEED" --lr "$LR" --checkpoint "$CKPT_ARG"
      --data_root ../../../dataset --domainbed_root ../../../domainbed_dataset
      "${NS_ARG[@]}" --output "../../../log/memo_${METHOD}_${DATASET}_s${SEED}")

# PROFILE=<n>: chi do thoi gian n anh dau (augmix CPU / fwd-bwd GPU) roi thoat, KHONG ghi ket qua.
if [ -n "${PROFILE:-}" ]; then
  python main.py "${ARGS[@]}" --profile "$PROFILE"
  exit 0
fi

# SHARDS=<k>: chia tap anh cua (dataset, corruption, seed) thanh k phan chay SONG SONG tren cung GPU roi gop lai.
# MEMO xu ly tung anh doc lap (nap lai checkpoint truoc moi anh) nen ket qua gop tuong duong chay 1 mach.
SHARDS=${SHARDS:-1}
if [ "$SHARDS" -gt 1 ]; then
  T=$(( $(nproc) / SHARDS )); [ "$T" -lt 1 ] && T=1
  export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
  # Chia shard luan phien qua cac GPU co san (NGPU=<n> de ghi de; khong co GPU/nvidia-smi -> 1)
  NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)}
  [ -z "$NGPU" ] || [ "$NGPU" -lt 1 ] && NGPU=1
  echo "MEMO: $SHARDS shard tren $NGPU GPU, $T luong CPU/shard"
  pids=()
  for ((i=0; i<SHARDS; i++)); do
    python main.py "${ARGS[@]}" --num_shards "$SHARDS" --shard_id "$i" --gpu_idx $((i % NGPU)) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python main.py "${ARGS[@]}" --merge_shards "$SHARDS"
else
  python main.py "${ARGS[@]}"
fi
