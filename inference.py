# infer_sd35_inpaint.py
import os, json, argparse, math, random
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from diffusers import StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as FM

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ----------------- utils -----------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def from_uint8(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = size_hw
    img = img.convert("RGB").resize((W, H), Image.BICUBIC)
    x = torch.from_numpy(np.array(img, dtype=np.float32))
    x = x.permute(2, 0, 1) / 255.0
    x = x * 2.0 - 1.0
    return x

def load_mask(p: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = size_hw
    m = Image.open(p).convert("L").resize((W, H), Image.NEAREST)
    t = torch.from_numpy(np.array(m, dtype=np.float32)) / 255.0
    t = (t > 0.5).float().unsqueeze(0)  # [1,H,W]
    return t

def denorm(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

# ----------------- dataset -----------------
class PairListDataset(torch.utils.data.Dataset):
    def __init__(self, list_file: str, size_hw: Tuple[int, int],
                 mask_based: bool = True, invert_mask: bool = True):
        super().__init__()
        self.items = []
        if list_file.endswith(".jsonl"):
            with open(list_file, "r") as f:
                for line in f:
                    self.items.append(json.loads(line))
        elif list_file.endswith(".json"):
            self.items = json.load(open(list_file))
        else:
            with open(list_file, "r") as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) == 2:
                        self.items.append({"person": parts[0], "garment": parts[1]})
                    elif len(parts) >= 3:
                        self.items.append({"person": parts[0], "garment": parts[1], "mask": parts[2]})
        self.H, self.W = size_hw
        self.mask_based = mask_based
        self.invert_mask = invert_mask

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        person = Image.open(meta["person"])
        garment = Image.open(meta["garment"])

        x_p = from_uint8(person, (self.H, self.W))   # [-1,1], [3,H,W]
        x_g = from_uint8(garment, (self.H, self.W))

        if self.mask_based:
            assert "mask" in meta, "mask_based=True requires mask path"
            m = load_mask(meta["mask"], (self.H, self.W))  # [1,H,W] 1=keep
            if self.invert_mask:
                m = 1.0 - m
            x_p_in = x_p * m
        else:
            m = torch.zeros(1, self.H, self.W, dtype=x_p.dtype)
            x_p_in = x_p

        x_concat_in = torch.cat([x_p_in, x_g], dim=2)      # [3,H,2W]
        x_concat_gt = torch.cat([x_p,   x_g], dim=2)       # [3,H,2W]
        m_concat    = torch.cat([m, torch.ones_like(m)], dim=2)  # [1,H,2W] right half=1
        return {"x_concat_in": x_concat_in, "x_concat_gt": x_concat_gt,
                "m_concat": m_concat, "meta": meta}

# ----------------- latent helpers -----------------
@torch.no_grad()
def to_latent_sd3(vae, x):
    vdtype = next(vae.parameters()).dtype
    posterior = vae.encode(x.to(vdtype)).latent_dist
    latents = posterior.sample() if torch.is_grad_enabled() else posterior.mean
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    return (latents - sh) * sf

@torch.no_grad()
def from_latent_sd3(vae, z: torch.Tensor) -> torch.Tensor:
    vdtype = next(vae.parameters()).dtype
    z = z.to(vdtype)
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    z = z / sf + sh
    img = vae.decode(z).sample
    return img

def fm_scale(x: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    # x / sqrt(sigma^2+1)
    if sigma_b.ndim == 1:
        sigma_b = sigma_b.view(-1, *([1] * (x.ndim - 1)))
    return (x.float() / torch.sqrt(sigma_b.float() ** 2 + 1.0)).to(x.dtype)

# unified conversion (raw->(eps,x0))
def raw_to_eps_x0(raw: torch.Tensor, z_t: torch.Tensor, sigma4: torch.Tensor, pred_type: str):
    pt = str(pred_type).lower()
    if pt == "epsilon":
        eps = raw.float()
        x0  = z_t.float() - sigma4.float() * eps
        return eps, x0
    if pt in ("x0", "sample"):
        x0  = raw.float()
        eps = (z_t.float() - x0) / sigma4.clamp_min(1e-8).float()
        return eps, x0
    raise NotImplementedError("v_prediction not supported here")

# ----------------- main infer class -----------------
class InpainterSD35:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(args.mixed_precision, torch.float16)

        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.sd3_model, torch_dtype=self.dtype, use_safetensors=True
        ).to(self.device)

        self.vae = pipe.vae
        self.transformer: SD3Transformer2DModel = pipe.transformer
        self.encode_prompt = pipe.encode_prompt
        self.scheduler = FM.from_config(pipe.scheduler.config)
        self.scheduler.config.prediction_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")

        # inject ckpt weights
        if args.ckpt_path and os.path.isfile(args.ckpt_path):
            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            sd = ckpt.get("transformer", ckpt)
            missing, unexpected = self.transformer.load_state_dict(sd, strict=False)
            print(f"[load] ckpt={args.ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[warn] ckpt not found or not provided — using base SD3.5 transformer weights.")

        # keep eval
        self.transformer.eval()

        # image size/vae scale
        self.H, self.W = args.size_h, args.size_w
        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # prompt cache
        self._null_pe  = None
        self._null_ppe = None

        # RNG
        self.gen = torch.Generator(device=self.device).manual_seed(args.seed)

    @torch.no_grad()
    def _encode_prompts(self, bsz: int):
        if self._null_pe is None:
            pe, _, ppe, _ = self.encode_prompt(
                prompt=[""], prompt_2=[""], prompt_3=[""],
                device=self.device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            self._null_pe  = pe.detach().to(self.dtype)
            self._null_ppe = ppe.detach().to(self.dtype)

        pe  = self._null_pe.expand(bsz, -1, -1).contiguous()
        ppe = self._null_ppe.expand(bsz, -1).contiguous()

        if self.args.zero_text:
            pe  = torch.zeros_like(pe)
            ppe = torch.zeros_like(ppe)
        return pe, ppe

    @torch.no_grad()
    def _set_fm_timesteps(self, num_steps: int):
        steps = max(2, int(num_steps))
        sched = FM.from_config(self.scheduler.config)
        # dynamic shifting (match training)
        tf_cfg = self.transformer.config
        patch = getattr(tf_cfg, "patch_size", 2)
        image_seq_len = (self.H // self.vae_scale // patch) * ((2 * self.W) // self.vae_scale // patch)
        if getattr(sched.config, "use_dynamic_shifting", False):
            base_len = getattr(sched.config, "base_image_seq_len", 256)
            max_len  = getattr(sched.config, "max_image_seq_len", 4096)
            base_s   = getattr(sched.config, "base_shift", 0.5)
            max_s    = getattr(sched.config, "max_shift", 1.16)
            mu = base_s + (max_s - base_s) * (image_seq_len - base_len) / max(1, (max_len - base_len))
            sched.set_timesteps(steps, device=self.device, mu=float(mu))
        else:
            sched.set_timesteps(steps, device=self.device)

        sigmas = sched.timesteps
        if sigmas[-1] != 0:
            sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        sched.sigmas = sigmas; sched.timesteps = sigmas
        return sched, sigmas

    @torch.no_grad()
    def infer_batch(self, batch: Dict[str, torch.Tensor], save_dir: str, base_name: str):
        x_in = batch["x_concat_in"].to(self.device, self.dtype)    # [-1,1] [B,3,H,2W]
        x_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m    = batch["m_concat"].to(self.device, self.dtype)       # [B,1,H,2W]

        B, _, Hh, WW = x_in.shape
        Ww = WW // 2

        # latents
        Xi = to_latent_sd3(self.vae, x_in).float()
        Mi = F.interpolate(m, size=(self.H // self.vae_scale, (2*self.W)//self.vae_scale), mode="nearest").float()
        K  = Mi

        # scheduler + start-step (preview_strength semantics)
        sched, sigmas_full = self._set_fm_timesteps(self.args.steps)
        s = max(0.0, min(1.0, float(self.args.strength)))
        N = sigmas_full.numel() - 1
        start_idx = min(max(int((1.0 - s) * N), 0), sigmas_full.numel() - 2)

        # sub-schedule
        sigmas = sigmas_full[start_idx:].contiguous()
        if sigmas[-1] != 0:
            sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        sched.sigmas = sigmas; sched.timesteps = sigmas

        # shared noise
        noise = torch.randn(Xi.shape, dtype=torch.float32, device=Xi.device, generator=self.gen)

        # initial masked z (what the model actually sees at step 0)
        s0 = sigmas[0].to(Xi.device, torch.float32)
        Xi_noisy0 = sched.scale_noise(Xi.float(), s0[None], noise).float()
        z = K * Xi_noisy0 + (1.0 - K) * (s0 * noise).float()

        # decode to pixel for "model input image"
        xi_noisy_img = from_latent_sd3(self.vae, z.to(self.dtype))
        xi_noisy_img = torch.clamp((xi_noisy_img + 1.0) * 0.5, 0.0, 1.0).detach()

        # prompts
        prompt_embeds, pooled = self._encode_prompts(B)

        # sampling loop with per-step recomposition
        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        self.transformer.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=self.dtype, enabled=(self.device.type=="cuda")):
            timesteps = sigmas
            for i in range(timesteps.numel() - 1):
                sigma_t   = timesteps[i].to(Xi.device, torch.float32)
                sigma_t_b = sigma_t.repeat(B)
                sigma4    = sigma_t.view(1,1,1,1).expand(B,1,1,1).float()

                z_in = fm_scale(z, sigma_t_b)

                raw = self.transformer(
                    hidden_states=z_in.to(self.dtype),
                    timestep=sigma_t_b,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled,
                    return_dict=True,
                ).sample.float()

                eps_pred, _ = raw_to_eps_x0(raw, z, sigma4, pred_type)
                z = sched.step(eps_pred, sigma_t, z).prev_sample.float()

                # per-step recomposition to enforce keep region
                sigma_next = timesteps[i + 1].to(Xi.device, torch.float32)
                init_latents_proper = sched.scale_noise(Xi.float(), sigma_next[None], noise).float()
                z = K * init_latents_proper + (1.0 - K) * z

        # decode and final pixel recomposition (keep mask from x_in)
        x_hat = from_latent_sd3(self.vae, z.to(self.dtype))
        K3 = m.repeat(1, 3, 1, 1)
        x_final = torch.clamp((K3 * x_in + (1.0 - K3) * x_hat + 1.0) * 0.5, 0.0, 1.0).detach()

        # build tiles in requested order
        person        = denorm(x_gt[:, :, :, :Ww])          # GPU
        garment       = denorm(x_gt[:, :, :, Ww:])          # GPU
        mask_keep     = m[:, :, :, :Ww]                     # GPU
        mask_vis      = mask_keep.repeat(1, 3, 1, 1)        # GPU
        masked_person = denorm(x_in[:, :, :, :Ww] * mask_keep)  # GPU
        model_input   = xi_noisy_img                         # GPU (이미 0..1)
        gt_pair       = denorm(x_gt)                         # GPU
        infer_out     = x_final                              # GPU (이미 0..1)

        # concat on GPU, then move once to CPU for saving
        cols  = [person, mask_vis, masked_person, garment, model_input, gt_pair, infer_out]
        panel = torch.cat(cols, dim=3).detach().cpu()        # <- 여기서만 CPU로 이동

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{base_name}.png")
        save_image(make_grid(panel, nrow=1, padding=0), out_path)
        return out_path

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--list_file", type=str, required=True, help="test list (json/jsonl/txt)")
    p.add_argument("--sd3_model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    p.add_argument("--ckpt_path", type=str, required=True, help="path to saved training ckpt")
    p.add_argument("--out_dir", type=str, default="infer_out")
    p.add_argument("--size_h", type=int, default=512)
    p.add_argument("--size_w", type=int, default=384)
    p.add_argument("--mask_based", action="store_true", help="set if your list has mask column")
    p.add_argument("--invert_mask", action="store_true", help="invert mask before use (CatVTON-style)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--strength", type=float, default=0.3, help="0→start near 0-noise, 1→full noise")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    p.add_argument("--zero_text", action="store_true", help="zero-out prompt embeddings to hard-disable cross path")
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    infer = InpainterSD35(args)

    ds = PairListDataset(args.list_file, size_hw=(args.size_h, args.size_w),
                         mask_based=args.mask_based, invert_mask=args.invert_mask)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs(args.out_dir, exist_ok=True)
    idx_base = 0
    for batch in tqdm(dl, desc="infer"):
        # make a name from indices or filenames
        metas = batch["meta"]
        names = []

        # 기본 collate_fn이면 dict of lists가 됩니다.
        if isinstance(metas, dict):
            persons  = metas.get("person", [])
            garments = metas.get("garment", [])
            B = len(persons)
            for i in range(B):
                stem_p = os.path.splitext(os.path.basename(persons[i]))[0]
                stem_g = os.path.splitext(os.path.basename(garments[i]))[0]
                names.append(f"{idx_base:06d}_{stem_p}__{stem_g}")
                idx_base += 1
        # 혹시 커스텀 collate_fn을 써서 list of dicts라면 여기로 처리
        else:
            for m in metas:
                stem_p = os.path.splitext(os.path.basename(m["person"]))[0]
                stem_g = os.path.splitext(os.path.basename(m["garment"]))[0]
                names.append(f"{idx_base:06d}_{stem_p}__{stem_g}")
                idx_base += 1

        # do single-batch inference and save per-sample panels
        # split batch to single items to name per sample
        B = batch["x_concat_in"].shape[0]
        for i in range(B):
            sub = {k: (v[i:i+1] if torch.is_tensor(v) else v) for k,v in batch.items() if k in ["x_concat_in","x_concat_gt","m_concat"]}
            out_path = infer.infer_batch(sub, args.out_dir, names[i])
            print(f"[save] {out_path}")

if __name__ == "__main__":
    main()