#!/bin/bash
# Chay nhanh 1 lenh voi --nsample nho de kiem tra loi shape/import truoc khi
# chay full. Luon chay script nay truoc khi chay run_all_bitta_family.sh.

set -e
cd "$(dirname "$0")/.."

python src/main.py --gpu_idx 0 --dataset cifar10 --method BiTTA --tgt cont \
    --model resnet18 --load_checkpoint_path pretrained_weights/cifar10/cp_last_0.pth.tar \
    --epoch 3 --seed 0 --remove_cp --online --tgt_train_dist 1 \
    --update_every_x 64 --memory_size 64 --memory_type ActivePriorityFIFO \
    --weight_decay 0 --lr 0.0001 --restoration_factor 0.0 \
    --use_learned_stats --bn_momentum 0.3 --dropout_rate 0.3 \
    --sample_selection info_max --n_dropouts_select 16 --explore_eps 0.0 --platt_min_samples 20 \
    --nsample 500 \
    --log_name log/ --log_prefix smoke_test

echo "Smoke test OK -- xoa log rac:"
echo "  rm -rf log/cifar10/BiTTA/tgt_cont/smoke_test"
