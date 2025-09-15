#!/usr/bin/env bash
set -euo pipefail

# ----- Required -----
DATA_CSV=${1:-"./DATA/VITON-HD/vitonhd_test_mask.csv"}
CKPT=${2:-"logs/20250903_165121_ddp/models/epoch_90_loss_0.2402.ckpt"}

# ----- Optional -----
OUTDIR=${OUTDIR:-"infer_out"}
SD3_MODEL=${SD3_MODEL:-"stabilityai/stable-diffusion-3.5-large"}
SIZE_H=${SIZE_H:-512}
SIZE_W=${SIZE_W:-384}
SEED=${SEED:-1234}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}   # fp16 | bf16 | fp32
INVERT_MASK=${INVERT_MASK:-1}                # 1 to invert
SAVE_PANEL=${SAVE_PANEL:-1}                  # 1 to save panels
NO_SAVE_CONCAT=${NO_SAVE_CONCAT:-0}          # 1 to skip saving concat
NO_SAVE_LEFT=${NO_SAVE_LEFT:-0}              # 1 to skip left-half
HF_TOKEN=${HF_TOKEN:-""}                     # if needed for gated repo
DEVICE=${DEVICE:-"auto"}                     # auto | cuda | cpu
OUT_SUB_FROM_CKPT=${OUT_SUB_FROM_CKPT:-1}    # 1 -> save under OUTDIR/<YYYYMMDD_HHMMSS>
DISABLE_TEXT=${DISABLE_TEXT:-1}              # 1 -> zero-text (match training), 0 -> encode empties

# Adapter / sampling knobs (match train defaults)
ADAPTER_ALPHA=${ADAPTER_ALPHA:-1.0}
STRENGTH=${STRENGTH:-0.60}                   # 0..1, img2img amount (0=keep input, 1=text2img-like)
STEPS=${STEPS:-32}
BS=${BS:-4}
NORM_MATCH_ADAPTER=${NORM_MATCH_ADAPTER:-1}  # 1 -> on (match train), 0 -> off
NORM_MATCH_CLIP=${NORM_MATCH_CLIP:-5.0}

# If CPU is requested, force fp32 for compatibility
if [[ "$DEVICE" == "cpu" ]]; then
  MIXED_PRECISION="fp32"
fi

EXTRA_ARGS=()
[[ "$INVERT_MASK" == "1" ]] && EXTRA_ARGS+=("--invert_mask")
[[ "$SAVE_PANEL" == "1" ]] && EXTRA_ARGS+=("--save_panel")
[[ "$NO_SAVE_CONCAT" == "1" ]] && EXTRA_ARGS+=("--no_save_concat")
[[ "$NO_SAVE_LEFT" == "1" ]] && EXTRA_ARGS+=("--no_save_left")
[[ -n "$HF_TOKEN" ]] && EXTRA_ARGS+=("--hf_token" "$HF_TOKEN")
[[ "$OUT_SUB_FROM_CKPT" == "1" ]] && EXTRA_ARGS+=("--out_sub_from_ckpt")
EXTRA_ARGS+=("--device" "$DEVICE")

# zero-text / text-embeds
if [[ "$DISABLE_TEXT" == "1" ]]; then
  EXTRA_ARGS+=("--disable_text")
else
  EXTRA_ARGS+=("--enable_text")
fi

# adapter & norm-match flags
EXTRA_ARGS+=("--adapter_alpha" "$ADAPTER_ALPHA")
EXTRA_ARGS+=("--strength" "$STRENGTH")
if [[ "$NORM_MATCH_ADAPTER" == "0" ]]; then
  EXTRA_ARGS+=("--no_norm_match_adapter")
fi
EXTRA_ARGS+=("--norm_match_clip" "$NORM_MATCH_CLIP")

python inference.py \
  --data_csv "$DATA_CSV" \
  --ckpt "$CKPT" \
  --sd3_model "$SD3_MODEL" \
  --outdir "$OUTDIR" \
  --size_h "$SIZE_H" \
  --size_w "$SIZE_W" \
  --steps "$STEPS" \
  --batch_size "$BS" \
  --seed "$SEED" \
  --mixed_precision "$MIXED_PRECISION" \
  "${EXTRA_ARGS[@]}"
