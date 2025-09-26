# train.py  (YAML-first, no dataclass) — DDP-ready, FlowMatch-correct, preview/save/log only on rank 0
import os
import json
import random
import argparse
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import itertools
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

try:
    import yaml
except Exception:
    yaml = None

# ------------------------------------------------------------
# Speed/precision knobs
# ------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------------------------------------------------------------
# Diffusers (SD3.5 / SD3)
# ------------------------------------------------------------
try:
    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
except Exception as e:
    raise ImportError(
        "This training script requires diffusers with SD3 support. "
        "Install/upg: pip install -U 'diffusers>=0.34.0' 'transformers>=4.41.0' 'safetensors>=0.4.3'"
    ) from e

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
except Exception as e:
    raise ImportError(
        "FlowMatchEulerDiscreteScheduler not found. Please update diffusers (>=0.29, preferably >=0.34)."
    ) from e

from huggingface_hub import snapshot_download


# ------------------------------------------------------------
# Small utils
# ------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def from_uint8(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    """PIL -> [-1,1] CHW float tensor"""
    img = img.convert("RGB").resize(size_hw[::-1], Image.BICUBIC)
    x = torch.from_numpy(np.array(img, dtype=np.float32))  # HWC, [0,255]
    x = x.permute(2, 0, 1) / 255.0                         # CHW, [0,1]
    x = x * 2.0 - 1.0                                      # [-1,1]
    return x


def load_mask(p: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Return (1,H,W) float mask in {0,1}"""
    m = Image.open(p).convert("L").resize(size_hw[::-1], Image.NEAREST)
    t = torch.from_numpy(np.array(m, dtype=np.float32)) / 255.0
    t = (t > 0.5).float().unsqueeze(0)
    return t


class DotDict(dict):
    """dict with attribute access (cfg.key)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# ---- DDP helpers ----
def maybe_init_distributed() -> Dict[str, int]:
    """
    Initialize distributed if env vars exist. Safe to call once (main).
    """
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return {"is_dist": True, "rank": rank, "world_size": world_size, "local_rank": local_rank}
    return {"is_dist": False, "rank": 0, "world_size": 1, "local_rank": 0}


def get_distributed_info() -> Dict[str, int]:
    """Read current DDP state without initializing."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return {"is_dist": True, "rank": rank, "world_size": world_size, "local_rank": local_rank}
    return {"is_dist": False, "rank": 0, "world_size": 1, "local_rank": 0}


def bcast_object(obj, src: int = 0):
    if dist.is_available() and dist.is_initialized():
        lst = [obj]
        dist.broadcast_object_list(lst, src=src)
        return lst[0]
    return obj


class NoopWriter:
    def add_scalar(self, *a, **k): pass
    def close(self): pass


def ddp_state_dict(m: nn.Module):
    return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class PairListDataset(Dataset):
    def __init__(self, list_file: str, size_hw: Tuple[int, int],
                 mask_based: bool = True, invert_mask: bool = False):
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        person = Image.open(meta["person"])
        garment = Image.open(meta["garment"])

        x_p = from_uint8(person, (self.H, self.W))
        x_g = from_uint8(garment, (self.H, self.W))

        if self.mask_based:
            assert "mask" in meta, "mask_based=True requires mask path per line"
            m = load_mask(meta["mask"], (self.H, self.W))  # 1 on foreground by dataset def
            if self.invert_mask:                           # if white=cloth(remove) -> convert to keep-mask
                m = 1.0 - m
            x_p_in = x_p * m                               # I_p ⊙ M_keep
        else:
            m = torch.zeros(1, self.H, self.W, dtype=x_p.dtype)
            x_p_in = x_p

        x_concat_in = torch.cat([x_p_in, x_g], dim=2)      # [3,H,2W]
        x_concat_gt = torch.cat([x_p, x_g], dim=2)         # [3,H,2W]
        m_concat = torch.cat([m, torch.zeros_like(m)], dim=2)  # [1,H,2W] (right half = 0)

        return {"x_concat_in": x_concat_in, "x_concat_gt": x_concat_gt, "m_concat": m_concat}


# ------------------------------------------------------------
# SD3.5 helpers
# ------------------------------------------------------------
def load_sd3_components(model_id: str, device, dtype, token: str = None, revision: str = None):
    """
    (Unused in DDP path below; left for reference)
    """
    local_dir = snapshot_download(
        repo_id=model_id, token=token, revision=revision,
        resume_download=True, local_files_only=False,
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        local_dir, torch_dtype=dtype, local_files_only=True, use_safetensors=True,
    ).to(device)

    vae = pipe.vae
    transformer: SD3Transformer2DModel = pipe.transformer
    encode_prompt = pipe.encode_prompt
    sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return vae, transformer, encode_prompt, sched


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
    """latents -> image in [-1,1] (B,3,H,2W)"""
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    z = z / sf + sh
    img = vae.decode(z).sample  # [-1,1]
    return img


class ConcatProjector(nn.Module):
    def __init__(self, in_ch=33, out_ch=16):
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


def freeze_all_but_attn_sd3(transformer: SD3Transformer2DModel):
    for p in transformer.parameters():
        p.requires_grad = False
    train_count = 0
    for name, module in transformer.named_modules():
        if ".attn" in name:
            for p in module.parameters():
                p.requires_grad = True
            train_count += sum(p.numel() for p in module.parameters())
    return train_count


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
class CatVTON_SD3_Trainer:
    def __init__(self, cfg: DotDict, run_dirs: Dict[str, str], cfg_yaml_to_save: Optional[Dict[str, Any]] = None):
        self.cfg = cfg

        # distributed info (do NOT init here)
        dinfo = get_distributed_info()
        self.is_dist = dinfo["is_dist"]
        self.rank = dinfo["rank"]
        self.world_size = dinfo["world_size"]
        self.local_rank = dinfo["local_rank"]
        self.is_main = (self.rank == 0)

        seed_everything(cfg.seed + self.rank)

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        mp2dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        self.dtype = mp2dtype.get(cfg.mixed_precision, torch.float16)

        # ---- logging dirs & writer ----
        self.run_dir = run_dirs["run_dir"]
        self.img_dir = run_dirs["images"]
        self.model_dir = run_dirs["models"]
        self.tb_dir = run_dirs["tb"]

        if self.is_main:
            for d in [self.img_dir, self.model_dir, self.tb_dir]:
                os.makedirs(d, exist_ok=True)

        self.logger = logging.getLogger(f"catvton_{os.path.basename(self.run_dir)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.is_main and not self.logger.handlers:
            self.log_path = os.path.join(self.run_dir, "log.txt")
            fh = logging.FileHandler(self.log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(fh)
        else:
            self.logger.addHandler(logging.NullHandler())

        self.tb = SummaryWriter(self.tb_dir) if self.is_main else NoopWriter()

        # save merged config.yaml (main only)
        if cfg_yaml_to_save is not None and yaml is not None and self.is_main:
            with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg_yaml_to_save, f, sort_keys=False)

        # Token
        env_token = (os.environ.get("HUGGINGFACE_TOKEN")
                     or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                     or os.environ.get("HF_TOKEN"))
        token = cfg.hf_token or env_token
        if token is None and self.is_main:
            msg = "[warn] No HF token found. If the repo is gated, set HUGGINGFACE_TOKEN or pass --hf_token."
            print(msg); self.logger.info(msg)

        # Load SD3.5 — rank0 downloads, others read cache
        if self.is_main:
            local_dir = snapshot_download(
                repo_id=cfg.sd3_model, token=token, revision=None,
                resume_download=True, local_files_only=False
            )
        if self.is_dist:
            dist.barrier()
        if not self.is_main:
            local_dir = snapshot_download(
                repo_id=cfg.sd3_model, token=token, revision=None,
                resume_download=True, local_files_only=True
            )

        pipe = StableDiffusion3Pipeline.from_pretrained(
            local_dir, torch_dtype=self.dtype, local_files_only=True, use_safetensors=True,
        ).to(self.device)

        self.vae = pipe.vae
        self.transformer: SD3Transformer2DModel = pipe.transformer
        self.encode_prompt = pipe.encode_prompt
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        if self.is_main:
            self.logger.info(f"Loaded SD3 model: {cfg.sd3_model}")

        # memory knobs (before DDP wrap)
        try:
            self.transformer.enable_gradient_checkpointing()
            msg = "[mem] gradient checkpointing ON"
            if self.is_main: print(msg)
            self.logger.info(msg)
        except Exception as e:
            msg = f"[mem] gradient checkpointing not available: {e}"
            if self.is_main: print(msg)
            self.logger.info(msg)
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            msg = "[mem] xFormers attention ON"
            if self.is_main: print(msg)
            self.logger.info(msg)
        except Exception as e:
            msg = f"[mem] xFormers not available: {e}"
            if self.is_main: print(msg)
            self.logger.info(msg)

        # projector: keep FP32
        self.projector = ConcatProjector(in_ch=33, out_ch=16).to(self.device, dtype=torch.float32)

        # freeze except attention (before DDP)
        trainable_tf = freeze_all_but_attn_sd3(self.transformer)

        # cast trainable(attn) params to FP32 when using fp16 training
        casted = 0
        if self.dtype == torch.float16:
            for _, p in self.transformer.named_parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)
                    casted += 1
        msg = f"[dtype] casted_trainable_to_fp32={casted}"
        if self.is_main: print(msg)
        self.logger.info(msg)

        proj_params = sum(p.numel() for p in self.projector.parameters())
        msg = f"Trainable params: transformer-attn={trainable_tf/1e6:.2f}M, projector={proj_params/1e6:.2f}M"
        if self.is_main: print(msg)
        self.logger.info(msg)

        # data
        self.dataset = PairListDataset(
            cfg.list_file, size_hw=(cfg.size_h, cfg.size_w),
            mask_based=cfg.mask_based, invert_mask=cfg.invert_mask
        )
        if self.is_dist:
            self.sampler = DistributedSampler(
                self.dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, drop_last=True
            )
        else:
            self.sampler = None

        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )
        self.logger.info(f"Dataset len={len(self.dataset)} batch_size(per-rank)={cfg.batch_size} invert_mask={cfg.invert_mask}")

        if self.is_dist:
            self.steps_per_epoch = max(1, self.sampler.num_samples // cfg.batch_size)
        else:
            self.steps_per_epoch = max(1, len(self.dataset) // cfg.batch_size)

        # AMP/GradScaler
        self.use_scaler = (self.device.type == "cuda") and (self.dtype == torch.float16)
        msg = f"[amp] dtype={self.dtype}, use_scaler={self.use_scaler}"
        if self.is_main: print(msg)
        self.logger.info(msg)

        # DDP wrap (after freezing/casting)
        self.transformer = DDP(
            self.transformer, device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, find_unused_parameters=False
        )
        self.projector = DDP(
            self.projector, device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, find_unused_parameters=False
        )

        # optimizer
        optim_params = itertools.chain(
            (p for p in self.transformer.parameters() if p.requires_grad),
            self.projector.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            optim_params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

        self._has_scale_model_input = hasattr(self.scheduler, "scale_model_input")

        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

        # preview noise generator (fixed for reproducibility)
        self._preview_gen = torch.Generator(device=self.device).manual_seed(cfg.preview_seed)

    def _encode_prompts(self, bsz: int):
        empties = [""] * bsz
        try:
            pe, _, ppe, _ = self.encode_prompt(
                prompt=empties, prompt_2=empties, prompt_3=empties,
                device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        except TypeError:
            pe, _, ppe, _ = self.encode_prompt(
                prompt=empties, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        return pe.to(self.dtype), ppe.to(self.dtype)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

    # --- timesteps helpers ---
    def _as_1d_timesteps(self, sample: torch.Tensor, t, B: int) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=sample.device)
        t = t.to(sample.device)
        if t.ndim == 0:
            t = t[None]
        tdtype = getattr(self.scheduler, "timesteps", torch.tensor([], device=sample.device)).dtype \
                if hasattr(self.scheduler, "timesteps") else torch.long
        t = t.to(tdtype)
        if t.shape[0] != B:
            t = t[:1].repeat(B)
        return t

    def _scale_model_input_safe(self, sample: torch.Tensor, timesteps) -> torch.Tensor:
        if not self._has_scale_model_input:
            return sample
        try:
            B = sample.shape[0]
            t1d = self._as_1d_timesteps(sample, timesteps, B)
            return self.scheduler.scale_model_input(sample, t1d)
        except Exception as e:
            self.logger.info(f"[scale_model_input_safe] bypass: {e}")
            return sample

    @torch.no_grad()
    def _preview_sample(self, batch: Dict[str, torch.Tensor], num_steps: int) -> torch.Tensor:
        H, W = self.cfg.size_h, self.cfg.size_w
        x_in = batch["x_concat_in"].to(self.device, self.dtype)
        m    = batch["m_concat"].to(self.device, self.dtype)

        Xi = to_latent_sd3(self.vae, x_in).to(self.dtype)
        Mi = F.interpolate(m, size=(H // 8, (2 * W) // 8), mode="nearest").to(self.dtype)
        B, C, h, w = Xi.shape

        # timesteps / init noise
        self.scheduler.set_timesteps(num_steps, device=self.device)
        sigma0 = float(getattr(self.scheduler, "init_noise_sigma",
                               self.scheduler.sigmas[0] if hasattr(self.scheduler, "sigmas") else 1.0))
        z = torch.randn((B, C, h, w), generator=self._preview_gen,
                        device=self.device, dtype=torch.float32) * sigma0
        z = z.to(self.dtype)

        prompt_embeds, pooled = self._encode_prompts(B)

        for t in self.scheduler.timesteps:
            t1d = self._as_1d_timesteps(z, t, B)  # (B,)
            z_in  = self._scale_model_input_safe(z,  t1d)
            Xi_in = self._scale_model_input_safe(Xi.float(), t1d)

            hidden = torch.cat([z_in.float(), Xi_in, Mi.float()], dim=1)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)

            proj = self.projector.module(hidden).to(self.dtype) if isinstance(self.projector, DDP) else self.projector(hidden).to(self.dtype)
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
                model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
                v = model(
                    hidden_states=proj,
                    timestep=t1d,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled,
                    return_dict=True,
                ).sample

            z = self.scheduler.step(v, t, z).prev_sample

        x_hat = from_latent_sd3(self.vae, z)                 # [-1,1]
        return torch.clamp((x_hat + 1.0) * 0.5, 0.0, 1.0)    # [0,1]

    @torch.no_grad()
    def _save_preview(self, batch: Dict[str, torch.Tensor], global_step: int, max_rows: int = 4):
        if not self.is_main:
            return
        x_in = batch["x_concat_in"][:max_rows].to(self.device)
        x_gt = batch["x_concat_gt"][:max_rows].to(self.device)
        m    = batch["m_concat"][:max_rows].to(self.device)

        pred_img = self._preview_sample(
            {k: v[:max_rows] for k, v in batch.items()}, num_steps=self.cfg.preview_infer_steps
        )  # [B,3,H,2W] in [0,1]

        B, C, H, WW = x_gt.shape
        W = WW // 2

        person  = x_gt[:, :, :, :W]
        garment = x_gt[:, :, :, W:]
        mask_keep = m[:, :, :, :W]
        mask_vis  = mask_keep.repeat(1, 3, 1, 1)
        masked_person = person * mask_keep

        rows = []
        for i in range(B):
            tiles = [
                self._denorm(person[i]),
                mask_vis[i],
                self._denorm(masked_person[i]),
                self._denorm(garment[i]),
                self._denorm(x_in[i]),
                pred_img[i],                     # 6th: inference
                self._denorm(x_gt[i]),          # 7th: GT
            ]
            row = torch.cat(tiles, dim=2)
            rows.append(row)
        panel = torch.stack(rows, dim=0)
        grid = make_grid(panel, nrow=1)
        out_path = os.path.join(self.img_dir, f"step_{global_step:06d}.png")
        save_image(grid, out_path)
        self.logger.info(f"[img] saved preview at step {global_step}: {out_path}")

    def step(self, batch, global_step: int):
        H, W = self.cfg.size_h, self.cfg.size_w

        x_concat_in = batch["x_concat_in"].to(self.device, self.dtype)
        x_concat_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m_concat = batch["m_concat"].to(self.device, self.dtype)

        with torch.no_grad():
            Xi = to_latent_sd3(self.vae, x_concat_in).to(self.dtype)           # [B,16,H/8,2W/8]
            z0 = to_latent_sd3(self.vae, x_concat_gt).to(self.dtype)           # [B,16,H/8,2W/8]
            Mi = F.interpolate(m_concat, size=(H // 8, (2 * W) // 8), mode="nearest").to(self.dtype)

        B = Xi.shape[0]

        if self.cfg.cond_dropout_p > 0:
            drop = (torch.rand(B, device=self.device, dtype=self.dtype) < self.cfg.cond_dropout_p).float().view(B, 1, 1, 1)
            Xi = Xi * (1.0 - drop)
            Mi = Mi * (1.0 - drop)

        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (B,),
            device=self.device, dtype=torch.long
        )

        # --------- Build noisy & target (FlowMatchEuler 정식) ---------
        z0_f32     = z0.float()
        noise_f32  = torch.randn_like(z0_f32)
        noisy_f32  = self.scheduler.add_noise(original_samples=z0_f32, noise=noise_f32, timesteps=timesteps)
        noisy_f32  = torch.nan_to_num(noisy_f32, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_type = getattr(self.scheduler.config, "prediction_type", "v_prediction")
        if pred_type == "epsilon":
            target_f32 = noise_f32
        elif pred_type == "v_prediction":
            target_f32 = self.scheduler.get_velocity(sample=z0_f32, noise=noise_f32, timesteps=timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        # --------- (safe) preconditioning ---------
        scaled_noisy_f32 = self._scale_model_input_safe(noisy_f32, timesteps)
        scaled_Xi_f32    = self._scale_model_input_safe(Xi.float(), timesteps)

        hidden_cat_f32 = torch.cat([scaled_noisy_f32, scaled_Xi_f32, Mi.float()], dim=1)
        hidden_cat_f32 = torch.nan_to_num(hidden_cat_f32, nan=0.0, posinf=30.0, neginf=-30.0)
        hidden_cat_f32 = torch.clamp(hidden_cat_f32, -30.0, 30.0)

        proj_out_f32 = (self.projector.module(hidden_cat_f32) if isinstance(self.projector, DDP)
                        else self.projector(hidden_cat_f32))                # FP32
        hidden_in    = proj_out_f32.to(self.dtype)

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(B)

        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
            model = self.transformer
            out = model(
                hidden_states=hidden_in,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            ).sample

        out_f32 = out.float()
        loss = F.mse_loss(out_f32, target_f32, reduction="mean")
        if not torch.isfinite(loss) or not loss.requires_grad:
            return None
        return loss

    def _save_ckpt(self, epoch: int, train_loss_epoch: float, global_step: int) -> str:
        ckpt_path = os.path.join(self.model_dir, f"[Train]_[{epoch}]_[{train_loss_epoch:.04f}].ckpt")
        payload = {
            "transformer": ddp_state_dict(self.transformer),
            "projector": ddp_state_dict(self.projector),
            "optimizer": self.optimizer.state_dict(),
            "cfg": dict(self.cfg),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_epoch": float(train_loss_epoch),
        }
        if self.is_main:
            torch.save(payload, ckpt_path)
            self.logger.info(f"[save] {ckpt_path}")
        return ckpt_path

    def train(self):
        global_step = 0
        epoch = 0
        self.transformer.train()
        self.projector.train()
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler)

        data_iter = itertools.cycle(self.loader)

        pbar = tqdm(
            total=self.cfg.max_steps,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}",
            leave=True,
            disable=(not self.is_main) or (not sys.stdout.isatty())
        )

        if self.is_dist and self.sampler is not None:
            self.sampler.set_epoch(epoch)

        while global_step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            nonfinite = False
            reason = ""

            for _ in range(self.cfg.grad_accum):
                batch = next(data_iter)
                loss = self.step(batch, global_step)

                if (loss is None) or (not torch.isfinite(loss)) or (not loss.requires_grad):
                    nonfinite = True
                    if loss is None:
                        reason = "None/NaN in forward"
                    elif not torch.isfinite(loss):
                        reason = "non-finite loss"
                    else:
                        reason = "detached loss"
                    break

                if self.use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += float(loss.detach().cpu())

            if nonfinite:
                msg = f"[warn] skipping step {global_step}: {reason}."
                if self.is_main:
                    pbar.write(msg)
                self.logger.info(msg)
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_scaler:
                    scaler.update()
                global_step += 1
                if self.is_main:
                    pbar.update(1)
                continue

            do_clip = True
            if self.use_scaler:
                try:
                    scaler.unscale_(self.optimizer)
                except ValueError as e:
                    msg = f"[amp] unscale_ skipped: {e}"
                    if self.is_main:
                        pbar.write(msg)
                    self.logger.info(msg)
                    do_clip = False

            if do_clip:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.transformer.parameters() if p.requires_grad), 0.5
                )
                torch.nn.utils.clip_grad_norm_(self.projector.parameters(), 0.5)

            if self.use_scaler:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            global_step += 1
            if self.is_main:
                pbar.update(1)

            self._epoch_loss_sum += loss_accum / max(1, self.cfg.grad_accum)
            self._epoch_loss_count += 1
            train_loss_avg = self._epoch_loss_sum / max(1, self._epoch_loss_count)

            if self.is_main and ((global_step % self.cfg.log_every) == 0 or global_step == 1):
                self.tb.add_scalar("train/loss", train_loss_avg, global_step)
                prog = (global_step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0.0
                pct = int(prog * 100)
                line = (f"Epoch {epoch}: {pct:3d}% | step {global_step}/{self.cfg.max_steps} "
                        f"| loss={train_loss_avg:.4f}")
                pbar.set_postfix_str(f"loss={train_loss_avg:.4f}")
                pbar.write(line)
                self.logger.info(line)

            if self.is_main and ((global_step % self.cfg.image_every) == 0 or global_step == 1):
                try:
                    self._save_preview(batch, global_step, max_rows=min(4, self.cfg.batch_size))
                    pbar.write(f"[img] saved preview at step {global_step}")
                except Exception as e:
                    msg = f"[warn] preview save failed: {e}"
                    pbar.write(msg); self.logger.info(msg)

            if (global_step % self.steps_per_epoch) == 0:
                epoch += 1
                if self.is_main and (epoch % self.cfg.save_epoch_ckpt) == 0:  # every N epochs
                    path = self._save_ckpt(epoch, train_loss_avg, global_step)
                    pbar.write(f"[save-epoch] {path}")
                self._epoch_loss_sum = 0.0
                self._epoch_loss_count = 0
                if self.is_main:
                    pbar.set_description(f"Epoch {epoch}")
                if self.is_dist and self.sampler is not None:
                    self.sampler.set_epoch(epoch)

        final_loss = self._epoch_loss_sum / max(1, self._epoch_loss_count) if self._epoch_loss_count > 0 else 0.0
        if self.is_main:
            path = self._save_ckpt(epoch, final_loss, global_step)
            pbar.write(f"[final] {path}")
        pbar.close()
        if self.is_main:
            self.tb.close()
        self.logger.info("[done] training finished")
        if self.is_dist:
            dist.barrier()


# ------------------------------------------------------------
# CLI / Config
# ------------------------------------------------------------
DEFAULTS = {
    # data / model
    "list_file": None,
    "sd3_model": "stabilityai/stable-diffusion-3.5-large",
    "size_h": 512, "size_w": 384,
    "mask_based": True, "invert_mask": False,

    # opt
    "lr": 1e-5, "batch_size": 4, "grad_accum": 1, "max_steps": 128000,
    "seed": 1337, "num_workers": 4, "cond_dropout_p": 0.1,
    "mixed_precision": "fp16", "loss_sigma_weight": False,

    # logging / save
    "save_root_dir": "logs", "save_name": "catvton_sd35",
    "log_every": 50, "image_every": 500, "save_every": 12800,
    "save_epoch_ckpt": 15,

    # preview sampling
    "preview_infer_steps": 16,
    "preview_seed": 1234,

    # hf
    "hf_token": None,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/catvton_sd35.yaml",
                   help="YAML config path. CLI overrides YAML.")
    # optional overrides
    p.add_argument("--list_file", type=str, default=None)
    p.add_argument("--sd3_model", type=str, default=None)
    p.add_argument("--size_h", type=int, default=None)
    p.add_argument("--size_w", type=int, default=None)
    p.add_argument("--mask_based", action="store_true")
    p.add_argument("--mask_free", action="store_true")
    p.add_argument("--invert_mask", action="store_true")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--image_every", type=int, default=None)
    p.add_argument("--cond_dropout_p", type=float, default=None)
    p.add_argument("--mixed_precision", type=str, default=None, choices=["fp16", "fp32", "bf16"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    p.add_argument("--save_root_dir", type=str, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--preview_infer_steps", type=int, default=None)
    p.add_argument("--preview_seed", type=int, default=None)
    return p.parse_args()


def load_merge_config(args: argparse.Namespace) -> DotDict:
    # start with defaults
    cfg = dict(DEFAULTS)

    # YAML layer
    if args.config and os.path.isfile(args.config):
        if yaml is None:
            raise RuntimeError("PyYAML not installed. `pip install pyyaml` or omit --config.")
        with open(args.config, "r") as f:
            y = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in y.items() if v is not None})

    # CLI overrides (truthy flags handled below)
    for k in list(cfg.keys()):
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None and not isinstance(v, bool):
                cfg[k] = v

    # exclusive flags
    if getattr(args, "mask_free", False):
        cfg["mask_based"] = False
    elif getattr(args, "mask_based", False):
        cfg["mask_based"] = True
    if getattr(args, "invert_mask", False):
        cfg["invert_mask"] = True

    if not cfg.get("list_file"):
        raise ValueError("`list_file` must be provided (YAML or CLI).")

    return DotDict(cfg)


def build_run_dirs(cfg: DotDict, run_name: Optional[str] = None, create: bool = True) -> Dict[str, str]:
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_{cfg.save_name}"
    run_dir = os.path.join(cfg.save_root_dir, run_name)
    paths = {
        "run_dir": run_dir,
        "images": os.path.join(run_dir, "images"),
        "models": os.path.join(run_dir, "models"),
        "tb": os.path.join(run_dir, "tb"),
    }
    if create:
        for d in paths.values():
            os.makedirs(d, exist_ok=True)
    return paths


def main():
    args = parse_args()
    cfg = load_merge_config(args)

    # init DDP once (main function)
    dinfo = maybe_init_distributed()
    is_dist = dinfo["is_dist"]
    rank = dinfo["rank"]

    # one shared run_name across ranks
    run_name = None
    if rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_{cfg.save_name}"
    run_name = bcast_object(run_name, src=0)
    run_dirs = build_run_dirs(cfg, run_name=run_name, create=(rank == 0))

    # save merged config only on rank 0
    cfg_yaml_to_save = dict(cfg) if rank == 0 else None

    trainer = CatVTON_SD3_Trainer(cfg, run_dirs, cfg_yaml_to_save=cfg_yaml_to_save)
    trainer.train()

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
