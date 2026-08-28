#!/bin/bash
# Cach dung: bash run_deyo_family.sh <method> <dset> <exp_type> <seed>
# Vi du:     bash run_deyo_family.sh deyo Waterbirds spurious 2024
#            bash run_deyo_family.sh deyo ImageNet-C mix_shifts 2024   # setting Mixed shift

set -e

METHOD=$1      # no_adapt | tent | eata | sar | deyo
DSET=$2        # ImageNet-C | Waterbirds | ColoredMNIST
EXP_TYPE=$3    # normal | mix_shifts | bs1 | label_shifts | spurious
SEED=${4:-2024}

cd "$(dirname "$0")/../src/methods/deyo"

CONTINUAL=False
if [ "$EXP_TYPE" != "normal" ] && [ "$EXP_TYPE" != "spurious" ]; then
  CONTINUAL=True
fi

python main.py \
    --method "$METHOD" --dset "$DSET" --exp_type "$EXP_TYPE" --seed "$SEED" \
    --continual "$CONTINUAL" \
    --model resnet18_bn \
    --aug_type patch --patch_len 4 \
    --deyo_margin 0.5 --deyo_margin_e0 0.4 --plpd_threshold 0.2 \
    --data_root ../../../dataset/ \
    --output ../../../log/deyo_${METHOD}_${DSET}_${EXP_TYPE}_${SEED}
