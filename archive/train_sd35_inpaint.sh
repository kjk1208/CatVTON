#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_sd35_inpaint.py \
  --config configs/sd35_inpaint.yaml \
  --mixed_precision bf16 \
  --save_name sd35_inpaint \