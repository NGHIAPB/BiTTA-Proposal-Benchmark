#!/bin/bash
# Chay kich ban cua DeYO va MEMO (2 phuong phap co runner rieng, xem run_deyo_family.sh / run_memo_family.sh).
#
#   DeYO : continuous (cifar10_c cifar100_c tiny_imagenet_c pacs) + mixed (cifar10_c pacs) + fully (7 dataset gom waterbirds)
#   MEMO : CHI fully (episodic: nap lai checkpoint truoc moi anh) tren 7 dataset gom waterbirds
#   Seed : pacs 0-4 (5 seed), con lai 0-2 (3 seed), giong run_all_bitta_family.sh.
#   MEMO chay tung (corruption/domain) -> 15 corruption x seed, rat lau: nen dat NSAMPLE (xem run_memo_family.sh).
#
# Bien moi truong: WHICH="deyo memo" (loc), SETTINGS="continuous mixed fully", DATASETS="...", NSAMPLE=<n> (chi cho MEMO).
# Vi du: WHICH=deyo SETTINGS=continuous DATASETS=cifar10_c bash run_all_other_methods.sh
#        WHICH=memo DATASETS=waterbirds bash run_all_other_methods.sh

set -e
cd "$(dirname "$0")"
WHICH=${WHICH:-"deyo memo"}
SETTINGS=${SETTINGS:-"continuous mixed fully"}
DATASETS=${DATASETS:-"cifar10_c cifar100_c tiny_imagenet_c pacs imagenet_r colored_mnist waterbirds"}
CORR15="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

valid() {
  case $1 in
    continuous) [[ "$2" =~ ^(cifar10_c|cifar100_c|tiny_imagenet_c|pacs)$ ]] ;;
    mixed)      [[ "$2" =~ ^(cifar10_c|pacs)$ ]] ;;
    fully)      return 0 ;;
  esac
}
seeds_of() { if [ "$1" = "pacs" ]; then echo "0 1 2 3 4"; else echo "0 1 2"; fi; }
memo_ds() { case $1 in cifar10_c) echo cifar10;; cifar100_c) echo cifar100;; tiny_imagenet_c) echo tiny-imagenet;; *) echo $1;; esac; }
memo_units() {
  case $1 in
    cifar10_c|cifar100_c|tiny_imagenet_c) echo $CORR15 ;;
    pacs) echo "art_painting cartoon sketch" ;;
    imagenet_r) echo corrupt ;;
    colored_mnist|waterbirds) echo test ;;
  esac
}

N=0
for setting in $SETTINGS; do
  for ds in $DATASETS; do
    valid "$setting" "$ds" || continue
    for s in $(seeds_of "$ds"); do
      if [[ " $WHICH " == *" deyo "* ]]; then
        N=$((N+1)); echo ">>> [$N] deyo | $ds | $setting | seed=$s"
        bash run_deyo_family.sh deyo "$ds" "$setting" "$s"
      fi
      if [[ " $WHICH " == *" memo "* ]] && [ "$setting" = "fully" ]; then
        for u in $(memo_units "$ds"); do
          N=$((N+1)); echo ">>> [$N] memo | $ds | fully | $u | seed=$s"
          bash run_memo_family.sh memo "$(memo_ds "$ds")" "$u" "$s" 5
        done
      fi
    done
  done
done
echo "Hoan tat $N luot. Tong hop: cd ../results && python aggregate_results.py --all"
