# WaterBirds

Chua co du lieu trong thu muc nay (chi la scaffold).

Benchmark tuong quan gia (Sagawa et al., 2020): anh chim (landbird / waterbird, 2 lop) ghep len nen dat / nen nuoc.
Train: 95% chim dat o nen dat, chim nuoc o nen nuoc; `val`/`test` can bang lai nen -> mo hinh "an gian" bam vao nen se sai.
Nhom = `2*y + place` (y: 0 landbird / 1 waterbird; place: 0 land / 1 water) -> LL / LS / SL / SS.

## Du lieu
Ban chinh thuc: `waterbird_complete95_forest2water2.tar.gz` (https://nlp.stanford.edu/data/dro/), giai nen thanh
```
dataset/WaterBirds/waterbird_complete95_forest2water2/
    metadata.csv            # img_id,img_filename,y,split,place,place_filename (split 0 train / 1 val / 2 test)
    001.Black_footed_Albatross/*.jpg ...
```
Loader `src/data_loader/WATERBIRDSDataset.py` cung tu nhan file `*.h5py` (dinh dang do `pretrain_waterbirds.py` cua DeYO tao)
neu co. Tren Kaggle: them tap du lieu WaterBirds vao Input hoac bat Internet roi tai/giai nen ve day.
Ca 7 phuong phap (TENT/EATA/SAR/BiTTA/BiTTA-Proposal/MEMO/DeYO) doc chung thu muc nay.

## Checkpoint (bat buoc, tu huan luyen, 1 file chung)
Huan luyen ERM tren `train`. **Chay tu thu muc goc project** (KHONG `cd src`):
```bash
python src/main.py --gpu_idx 0 --dataset waterbirds --method Src --src train --tgt test \
    --model resnet18_pretrained --epoch 20 --seed 0 --log_name log/ --log_prefix pretrain_wb
cp log/waterbirds/Src/src_train/tgt_test/pretrain_wb/cp/cp_last.pth.tar pretrained_weights/waterbirds/cp_last.pth.tar
```
KHONG truyen `--lr`. So epoch la vi du. Chi can 1 checkpoint (dung chung moi seed, nhu PACS).

## Setting
Chi co 1 domain dich (`test`) -> **chi Fully TTA**, voi ca 7 phuong phap:
```bash
bash scripts/run_bitta_family.sh bitta_proposal waterbirds fully 0
bash scripts/run_deyo_family.sh  deyo           waterbirds fully 0
bash scripts/run_memo_family.sh  memo           waterbirds test  0
```
Bao cao: do chinh xac theo 4 nhom (LL/LS/SL/SS), trung binh nhom (`avg_group`) va nhom te nhat (`worst_group`).
