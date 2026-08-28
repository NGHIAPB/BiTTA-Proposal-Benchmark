# WaterBirds

Chua co du lieu trong thu muc nay (chi la scaffold).

## Ban chat dataset
Benchmark **tuong quan gia (spurious correlation)**: 95% anh chim nuoc nam tren nen
nuoc, 95% anh chim dat nam tren nen dat (o tap huan luyen); ti le nay bi DAO NGUOC
o tap kiem thu. Dung de kiem tra mo hinh co dang "an gian" bam vao nen anh thay vi
hinh dang con chim hay khong — xem phan phan tich CPR/TRAP factors trong tai lieu
paper DeYO da tom tat.

## Nguon du lieu goc
Ket hop object tu CUB-200-2011 (Caltech-UCSD Birds) voi nen tu dataset Places.
Loader chinh thuc da co san tai:
    src/methods/deyo/dataset/waterbirds_dataset.py

## Trang thai tich hop trong project nay
- **DeYO**: da noi san day du (runner `src/methods/deyo/main.py --dset Waterbirds`).
- **BiTTA-family** (`src/main.py`): CHUA co loader rieng. Muon mo rong, xem huong dan
  "Them dataset moi cho BiTTA-family" trong README.md o thu muc goc project.
