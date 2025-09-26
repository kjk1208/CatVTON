python make_vitonhd_csv.py \
  --root DATA/VITON-HD \
  --split test \
  --out vitonhd_test_mask.csv \
  --mode mask \
  --check_mask

#python make_vitonhd_csv.py --root DATA/VITON-HD --split train --out vitonhd_train_mask.csv --mode mask --fallback_to_v3 # Stage-1(마스크 기반, agnostic-mask가 없으면 v3.2로 폴백 허용)
#python make_vitonhd_csv.py --root DATA/VITON-HD --split train --out vitonhd_train_maskfree.csv --mode maskfree # Stage-2(마스크 프리)
