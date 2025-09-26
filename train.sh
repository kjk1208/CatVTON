#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc || true
conda activate catvton

python train.py --config configs/catvton_sd35.yaml --mixed_precision bf16 --save_name myrun
