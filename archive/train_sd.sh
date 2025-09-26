#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_sd.py \
  --config configs/sd35.yaml \
  --mixed_precision bf16 \
  --save_name FFN_IO_QKV \