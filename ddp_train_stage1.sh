#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

torchrun --standalone --nnodes=1 --nproc_per_node=2 ddp_train_stage1.py \
  --config configs/stage1.yaml \
  --mixed_precision bf16 \
  --save_name scheduler_FM \
  --no_prefer_xformers \
  --strip_cross_attention \