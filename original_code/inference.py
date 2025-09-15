# inference.py — FlowMatch SD3.5 inference for CatVTON
# - matches ddp_train_without_crossattn.py:
#   * FlowMatchEulerDiscreteScheduler, prediction_type=v_prediction
#   * zero-text (default), no cross-attn conditioning
#   * z_t-only scaling (scale_model_input on noisy sample only)
#   * residual adapter A([Xi, Mi]) with mask gate (+ optional norm-match)
#   * img2img-style start using --strength (0..1)

import os
import re
import csv
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

try:
    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
except Exception as e:
    raise ImportError(
        "This script requires diffusers with SD3 support. "
        "Install: pip install -U 'diffusers>=0.34.0' 'transformers>=4.41.0' 'safetensors>=0.4.3'"
    ) from e

from huggingface_hub import snapshot_download
from tqdm import tqdm


# ---------- Utils (identical semantics to train) ----------
def seed_everything(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def from_uint8(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    img = img.convert("RGB").resize(size_hw[::-1], Image.BICUBIC)
    x = torch.from_numpy(np.array(img, dtype=np.float32))  # HWC
    x = x.permute(2, 0, 1) / 255.0                         # CHW
    x = x * 2.0 - 1.0
    return x


def load_mask(p: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    m = Image.open(p).convert("L").resize(size_hw[::-1], Image.NEAREST)
    t = torch.from_numpy(np.array(m, dtype=np.float32)) / 255.0
    t = (t > 0.5).float().unsqueeze(0)  # (1,H,W)
    return t


@torch.no_grad()
def to_latent_sd3(vae, x_bchw: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(x_bchw).latent_dist
    latents = posterior.sample()
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    latents = (latents - sh) * sf
    return latents


@torch.no_grad()
def from_latent_sd3(vae, z: torch.Tensor) -> torch.Tensor:
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    z = z / sf + sh
    img = vae.decode(z).sample  # [-1,1]
    return img


class ConcatProjector(nn.Module):
    # 16 (Xi) + 1 (Mi) = 17 -> 16 (adapter A)
    def __init__(self, in_ch=17, out_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1, bias=True),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)


def _as_1d_timesteps(scheduler, sample: torch.Tensor, t, B: int) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=sample.device)
    t = t.to(sample.device)
    if t.ndim == 0:
        t = t[None]
    tdtype = getattr(scheduler, "timesteps", torch.tensor([], device=sample.device)).dtype \
             if hasattr(scheduler, "timesteps") else torch.long
    t = t.to(tdtype)
    if t.shape[0] != B:
        t = t[:1].repeat(B)
    return t


def _scale_model_input_safe(scheduler, sample: torch.Tensor, timesteps) -> torch.Tensor:
    if not hasattr(scheduler, "scale_model_input"):
        return sample
    try:
        B = sample.shape[0]
        t1d = _as_1d_timesteps(scheduler, sample, timesteps, B)
        return scheduler.scale_model_input(sample, t1d)
    except Exception:
        return sample


def _denorm01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


# ---------- Dataset for CSV (image, cloth, agnostic-mask) ----------
class CSVVitonDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, size_h: int, size_w: int, invert_mask: bool = False):
        super().__init__()
        self.items: List[Dict[str, str]] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # expected headers: image, cloth, agnostic-mask
            if not set(["image", "cloth", "agnostic-mask"]).issubset(set(reader.fieldnames or [])):
                raise ValueError("CSV must have headers: image, cloth, agnostic-mask")
            for row in reader:
                self.items.append({
                    "person": row["image"],
                    "garment": row["cloth"],
                    "mask": row["agnostic-mask"],
                })
        self.H, self.W = size_h, size_w
        self.invert_mask = invert_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        person = Image.open(meta["person"])
        garment = Image.open(meta["garment"])
        mask_p = meta["mask"]

        x_p = from_uint8(person, (self.H, self.W))
        x_g = from_uint8(garment, (self.H, self.W))

        m = load_mask(mask_p, (self.H, self.W))
        if self.invert_mask:
            m = 1.0 - m
        x_p_in = x_p * m

        x_concat_in = torch.cat([x_p_in, x_g], dim=2)  # [3,H,2W]
        return {
            "x_p": x_p, "x_g": x_g, "m": m,
            "x_concat_in": x_concat_in,
            "paths": meta,  # keep raw paths for naming
        }


# ---------- helpers ----------
def _cfg_get(cfg, key, default=None):
    try:
        return getattr(cfg, key)
    except Exception:
        pass
    try:
        return cfg.get(key, default)
    except Exception:
        pass
    try:
        return cfg[key]
    except Exception:
        return default


def _derive_subdir_from_ckpt(ckpt_path: str) -> str:
    parent2 = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))  # run dir
    m = re.search(r"(\d{8}_\d{6})", parent2)
    return m.group(1) if m else parent2


# ---------- Inference Runner ----------
def run_inference(
    data_csv: str,
    ckpt_path: str,
    sd3_model: str,
    outdir: str,
    size_h: int,
    size_w: int,
    steps: int,
    batch_size: int,
    seed: int,
    dtype_str: str,
    invert_mask: bool,
    save_panel: bool,
    save_concat: bool,
    save_left: bool,
    hf_token: str = None,
    device_pref: str = "auto",
    out_sub_from_ckpt: bool = True,   # default on (convenience)
    adapter_alpha: float = 1.0,
    disable_text: bool = True,
    strength: float = 0.6,            # match train preview default
    norm_match_adapter: bool = True,
    norm_match_clip: float = 5.0,
):
    os.makedirs(outdir, exist_ok=True)

    if out_sub_from_ckpt:
        sub = _derive_subdir_from_ckpt(ckpt_path)
        outdir = os.path.join(outdir, sub)
        os.makedirs(outdir, exist_ok=True)
        print(f"[out] saving to {outdir}")

    # Resolve device
    device_pref = device_pref.lower()
    if device_pref not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Invalid --device '{device_pref}'. Choose from auto|cuda|cpu.")
    if device_pref == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision with CPU guard
    mp2dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = mp2dtype.get(dtype_str, torch.float16)
    if device.type == "cpu":
        if dtype != torch.float32:
            print("[warn] CPU detected → forcing mixed_precision=fp32 for compatibility.")
        dtype = torch.float32

    seed_everything(seed)

    # SD3.5 weights
    local_dir = snapshot_download(
        repo_id=sd3_model, token=hf_token, revision=None,
        resume_download=True, local_files_only=False
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        local_dir, torch_dtype=dtype, local_files_only=True, use_safetensors=True,
    ).to(device)

    if device.type == "cuda":
        try:
            pipe.transformer.enable_xformers_memory_efficient_attention()
            print("[info] xFormers memory-efficient attention enabled.")
        except Exception:
            print("[info] xFormers not available; using standard attention.]")

    vae = pipe.vae
    transformer: SD3Transformer2DModel = pipe.transformer
    encode_prompt = pipe.encode_prompt

    # ----- load ckpt & scheduler config -----
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")

    sched_cfg = payload.get("scheduler_config", None)
    if sched_cfg is not None:
        scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)
        print("[sched] restored scheduler_config from ckpt")
    else:
        scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        print("[sched] using scheduler from pipeline")

    # ensure v_prediction
    pred_type = _cfg_get(scheduler.config, "prediction_type", None)
    if pred_type != "v_prediction":
        print("[fix] overriding scheduler.prediction_type -> v_prediction")
        scheduler.register_to_config(prediction_type="v_prediction")

    # ----- load model weights -----
    msg_missing, msg_unexp = transformer.load_state_dict(payload["transformer"], strict=False)
    print(f"[load] transformer missing={len(msg_missing)} unexpected={len(msg_unexp)}")

    projector = ConcatProjector(in_ch=17, out_ch=16).to(device, dtype=torch.float32)
    proj_sd = payload.get("projector", None)
    if proj_sd is not None:
        cur = projector.state_dict()
        filt = {k: v for k, v in proj_sd.items() if (k in cur and cur[k].shape == v.shape)}
        miss, unexp = projector.load_state_dict(filt, strict=False)
        print(f"[load] projector loaded {len(filt)}/{len(cur)} keys "
              f"(missing={len(miss)} unexpected={len(unexp)})")
    else:
        print("[load] no projector weights in ckpt; using randomly initialized adapter.")

    transformer.eval()
    projector.eval()

    # Dataset & Loader
    dataset = CSVVitonDataset(data_csv, size_h=size_h, size_w=size_w, invert_mask=invert_mask)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # Prompt embeds helper — zero-text default
    def _encode_prompts(bsz: int, disable: bool):
        if disable:
            if not hasattr(_encode_prompts, "_pe_shape"):
                try:
                    pe_probe, _, ppe_probe, _ = encode_prompt(
                        prompt=[""], prompt_2=[""], prompt_3=[""],
                        device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
                    )
                except TypeError:
                    pe_probe, _, ppe_probe, _ = encode_prompt(
                        prompt=[""], device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
                    )
                _encode_prompts._pe_shape  = (pe_probe.shape[1], pe_probe.shape[2])
                _encode_prompts._ppe_shape = (ppe_probe.shape[1],)
            pe  = torch.zeros((bsz, *_encode_prompts._pe_shape),  dtype=dtype, device=device)
            ppe = torch.zeros((bsz, *_encode_prompts._ppe_shape), dtype=dtype, device=device)
            return pe, ppe
        else:
            empties = [""] * bsz
            try:
                pe, _, ppe, _ = encode_prompt(
                    prompt=empties, prompt_2=empties, prompt_3=empties,
                    device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
                )
            except TypeError:
                pe, _, ppe, _ = encode_prompt(
                    prompt=empties, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
                )
            return pe.to(dtype), ppe.to(dtype)

    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        pbar_batches = tqdm(loader, desc="Batches", ncols=120)
        for bi, batch in enumerate(pbar_batches):
            x_p = batch["x_p"].to(device)                 # [-1,1]
            x_g = batch["x_g"].to(device)
            m   = batch["m"].to(device)
            x_in= batch["x_concat_in"].to(device, dtype)  # [-1,1]

            B, _, H, WW = x_in.shape
            W = WW // 2

            # m_concat has left-half only; right-half zeros (matches train)
            m_concat = torch.cat([m, torch.zeros_like(m)], dim=2)  # [B,1,H,2W]

            # Latents & mask downsample
            Xi = to_latent_sd3(vae, x_in).to(dtype)                           # [B,16,H/8,2W/8]
            Mi = F.interpolate(m_concat, size=(H // 8, (2 * W) // 8), mode="nearest").to(dtype)

            # --- Sampler setup (img2img-style start) ---
            scheduler.set_timesteps(steps, device=device)
            s = float(max(0.0, min(1.0, strength)))
            T = len(scheduler.timesteps)
            init_timestep = min(int(T * s), T)
            t_start = max(T - init_timestep, 0)
            timesteps = scheduler.timesteps[t_start:]
            if timesteps.numel() == 0:
                timesteps = scheduler.timesteps[-1:].clone()
                t_start = T - 1

            noise = torch.randn(Xi.shape, generator=gen, dtype=torch.float32, device=device)

            try:
                z = scheduler.add_noise(Xi.float(), noise, timesteps[0]).to(dtype)
            except Exception:
                if hasattr(scheduler, "sigmas"):
                    sigma = scheduler.sigmas[t_start].to(device)
                else:
                    sigma = torch.tensor(float(getattr(scheduler, "init_noise_sigma", 1.0)),
                                         device=device, dtype=torch.float32)
                z = (Xi.float() + noise * sigma).to(dtype)

            prompt_embeds, pooled = _encode_prompts(B, disable_text)

            # Per-step loop (partial denoise only)
            pbar_steps = tqdm(timesteps, total=len(timesteps),
                              desc=f"Sampling b{bi}", ncols=120, leave=False)
            for t in pbar_steps:
                t1d  = _as_1d_timesteps(scheduler, z, t, B)
                z_in = _scale_model_input_safe(scheduler, z, t1d)             # ★ z_t만 스케일

                # adapter: A([Xi,Mi]) with mask gate (+ optional norm-match)
                cm    = torch.cat([Xi.float(), Mi.float()], dim=1)            # [B,17,h,w]
                cm    = torch.nan_to_num(cm, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
                adapt = projector(cm)                                         # [B,16,h,w] (fp32)
                gate  = (Mi > 0).float()
                adapt = adapt * gate

                if norm_match_adapter:
                    eps = 1e-6
                    num = z_in.float().flatten(1).norm(dim=1, keepdim=True)    # ||z_t||
                    den = adapt.flatten(1).norm(dim=1, keepdim=True) + eps     # ||A||
                    scale = (num / den).view(B, 1, 1, 1).clamp(0.0, float(norm_match_clip))
                    adapt = adapt * scale

                hidden_in = (z_in.float() + adapter_alpha * adapt).to(dtype)   # residual

                if device.type == "cuda":
                    with torch.amp.autocast(
                        device_type='cuda',
                        dtype=dtype,
                        enabled=(dtype in (torch.float16, torch.bfloat16))
                    ):
                        v = transformer(
                            hidden_states=hidden_in,
                            timestep=t1d,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled,
                            return_dict=True,
                        ).sample
                else:
                    v = transformer(
                        hidden_states=hidden_in,
                        timestep=t1d,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample

                z = scheduler.step(v, t, z).prev_sample

            x_hat = from_latent_sd3(vae, z)        # [-1,1], [B,3,H,2W]
            x_hat01 = _denorm01(x_hat)             # [0,1]

            # Saves
            for i in range(B):
                meta_i = batch["paths"][i] if isinstance(batch["paths"], list) else batch["paths"]
                base = os.path.splitext(os.path.basename(meta_i["person"]))[0]

                if save_concat:
                    out_concat_path = os.path.join(outdir, f"{base}_pred_concat.png")
                    save_image(x_hat01[i], out_concat_path)

                if save_left:
                    left = x_hat01[i, :, :, :W]
                    out_left_path = os.path.join(outdir, f"{base}_tryon.png")
                    save_image(left, out_left_path)

                if save_panel:
                    person = _denorm01(x_p[i])
                    garment = _denorm01(x_g[i])
                    mask_vis = m[i].repeat(3, 1, 1)
                    masked_person = person * mask_vis
                    concat_in = _denorm01(x_in[i])

                    tiles = [person, mask_vis, masked_person, garment, concat_in, x_hat01[i]]
                    row = torch.cat(tiles, dim=2)
                    grid = make_grid(row, nrow=1)
                    out_panel_path = os.path.join(outdir, f"{base}_panel.png")
                    save_image(grid, out_panel_path)

            print(f"[{bi+1}/{len(loader)}] saved batch")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, required=True,
                   help="CSV path with headers: image, cloth, agnostic-mask")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path, e.g., logs/.../epoch_xxx.ckpt")
    p.add_argument("--sd3_model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    p.add_argument("--outdir", type=str, default="infer_out")
    p.add_argument("--size_h", type=int, default=512)
    p.add_argument("--size_w", type=int, default=384)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--invert_mask", action="store_true", help="Invert keep-mask if dataset uses cloth region=white")
    p.add_argument("--save_panel", action="store_true")
    p.add_argument("--no_save_concat", action="store_true")
    p.add_argument("--no_save_left", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                   help="Select device. 'auto' uses CUDA if available, else CPU.")
    p.add_argument("--out_sub_from_ckpt", action="store_true",
                   help="Nest outputs under OUTDIR/<YYYYMMDD_HHMMSS> derived from CKPT path.")

    # NEW: align with train preview/adapter options
    p.add_argument("--adapter_alpha", type=float, default=1.0)
    p.add_argument("--disable_text", action="store_true")
    p.add_argument("--enable_text", action="store_true")
    p.add_argument("--strength", type=float, default=0.6,
                   help="0..1, img2img noise amount (0=keep input, 1=text2img-like)")
    p.add_argument("--no_norm_match_adapter", action="store_true",
                   help="Disable norm matching of adapter to z_t norm")
    p.add_argument("--norm_match_clip", type=float, default=5.0,
                   help="Clamp for norm-match scale (default 5.0)")
    return p.parse_args()


def main():
    args = parse_args()

    # resolve disable_text like train (default True unless --enable_text)
    disable_text = True
    if args.enable_text:
        disable_text = False
    elif args.disable_text:
        disable_text = True

    run_inference(
        data_csv=args.data_csv,
        ckpt_path=args.ckpt,
        sd3_model=args.sd3_model,
        outdir=args.outdir,
        size_h=args.size_h,
        size_w=args.size_w,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        dtype_str=args.mixed_precision,
        invert_mask=args.invert_mask,
        save_panel=args.save_panel,
        save_concat=(not args.no_save_concat),
        save_left=(not args.no_save_left),
        hf_token=args.hf_token,
        device_pref=args.device,
        out_sub_from_ckpt=args.out_sub_from_ckpt or True,  # keep default-on behavior
        adapter_alpha=args.adapter_alpha,
        disable_text=disable_text,
        strength=args.strength,
        norm_match_adapter=(not args.no_norm_match_adapter),
        norm_match_clip=args.norm_match_clip,
    )


if __name__ == "__main__":
    main()
