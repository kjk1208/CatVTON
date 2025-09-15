#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

torchrun --nproc_per_node=2 ddp_train.py \
 --config configs/catvton_sd35.yaml \
 --mixed_precision bf16 \
 --save_name ddp \
 --cond_dropout_p 0.1 \
