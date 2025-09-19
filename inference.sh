CUDA_VISIBLE_DEVICES=1 python inference.py \
  --list_file ./DATA/VITON-HD/vitonhd_test_mask.csv \
  --sd3_model stabilityai/stable-diffusion-3.5-large \
  --ckpt_path logs/save_all/20250916_012238_sd35_public_inpaint/models/epoch_25_loss_65.0764.ckpt \
  --out_dir infer_out \
  --mask_based --invert_mask \
  --steps 60 --strength 0.3 --seed 1234 \
  --mixed_precision bf16