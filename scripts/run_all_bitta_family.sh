#!/bin/bash
# Chay TOAN BO kich ban thuc nghiem = 210 run (moi run = 1 to hop method x dataset x setting x seed):
#
#   Setting     Dataset                                            Seed        # run
#   continuous  cifar10_c cifar100_c tiny_imagenet_c | pacs        3 | 5       (3+3+3+5) x 5 = 70
#   mixed       cifar10_c | pacs                                   3 | 5       (3+5) x 5     = 40
#   fully       cifar10_c cifar100_c tiny_imagenet_c imagenet_r
#               colored_mnist | pacs                               3 | 5       (3+3+3+3+3+5) x 5 = 100
#   Method: tent eata sar (co binary feedback) bitta_baseline bitta_proposal.
#
# Dieu kien: da co du lieu (dataset/, domainbed_dataset/) va checkpoint (pretrained_weights/), xem README.md.
# Bien moi truong: SETTINGS="continuous fully" (loc setting), DATASETS="pacs cifar10_c" (loc dataset),
#                  SKIP_DONE=1 (bo qua run da co online_eval.json de chay tiep sau khi bi ngat).
# Rat nang: nen chay tung setting/dataset mot (vd SETTINGS=continuous DATASETS=cifar10_c bash run_all_bitta_family.sh).

set -e
cd "$(dirname "$0")"

METHODS=(tent eata sar bitta_baseline bitta_proposal)
SETTINGS=${SETTINGS:-"continuous mixed fully"}
DATASETS=${DATASETS:-"cifar10_c cifar100_c tiny_imagenet_c pacs imagenet_r colored_mnist"}

valid() {   # $1 = setting, $2 = dataset
  case $1 in
    continuous) [[ "$2" =~ ^(cifar10_c|cifar100_c|tiny_imagenet_c|pacs)$ ]] ;;
    mixed)      [[ "$2" =~ ^(cifar10_c|pacs)$ ]] ;;
    fully)      return 0 ;;
  esac
}
seeds_of() { if [ "$1" = "pacs" ]; then echo "0 1 2 3 4"; else echo "0 1 2"; fi; }
done_marker() {   # ten thu muc log cua --dataset ung voi ten dataset trong script
  case $1 in cifar10_c) echo cifar10;; cifar100_c) echo cifar100;; tiny_imagenet_c) echo tiny-imagenet;;
             pacs) echo pacs;; imagenet_r) echo imagenetR;; colored_mnist) echo colored-mnist;; esac
}
method_dir() { case $1 in tent) echo TENT;; eata) echo EATA;; sar) echo SAR;; *) echo BiTTA;; esac; }

N=0
for setting in $SETTINGS; do
  for ds in $DATASETS; do
    valid "$setting" "$ds" || continue
    for m in "${METHODS[@]}"; do
      for s in $(seeds_of "$ds"); do
        N=$((N+1))
        prefix="${m}_${ds}_${setting}_s${s}"
        if [ -n "${SKIP_DONE:-}" ]; then
          d="../log/$(done_marker "$ds")/$(method_dir "$m")"
          if compgen -G "$d/tgt_*/$prefix/online_eval.json" >/dev/null || compgen -G "$d/tgt_*/$prefix/*/online_eval.json" >/dev/null; then
            echo ">>> [$N] bo qua (da xong): $prefix"; continue
          fi
        fi
        echo ">>> [$N] $m | $ds | $setting | seed=$s"
        bash run_bitta_family.sh "$m" "$ds" "$setting" "$s"
      done
    done
  done
done

echo "Hoan tat $N run. Tong hop: cd ../results && python aggregate_results.py --all"
