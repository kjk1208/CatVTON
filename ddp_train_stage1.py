# ddp_train_stage1.py — SD3.5 CatVTON (DDP-ready, 33ch concat path, patched for true inpainting)
# - Train: FlowMatch (ε-target) + masked latent x0 reconstruction
# - Infer/Preview: FlowMatch Euler (ε) + per-step recomposition (+ optional inpaint-start)
# - Zero-text embeds; freeze only self-attn Q/K/V/Out
# - 33ch concat: [z_t(16), Xi(16), Mi(1)] → projector(33→16) → transformer
# - HOLE-only warmup → hole-weighted loss (+ masked latent recon, + keep-consistency)
# - Adapter alpha warmup, cond-dropout warmup, projector HOLE-gate warmup
# - Param groups: transformer lr vs projector lr

import os
import re
import json
import random
import argparse
from typing import Tuple, Optional, Dict, Any, List
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

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

try:
    import yaml
except Exception:
    yaml = None

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    from diffusers import StableDiffusion3Pipeline
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
except Exception as e:
    raise ImportError(
        "This training script requires diffusers with SD3 support. "
        "Install/upg: pip install -U 'diffusers>=0.34.0' 'transformers>=4.41.0' 'safetensors>=0.4.3'"
    ) from e

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as FM
except Exception as e:
    raise ImportError("Schedulers not found. Please update diffusers (>=0.34).") from e

try:
    from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
    _HAS_ATTENTION_PROCESSORS = True
except Exception:
    _HAS_ATTENTION_PROCESSORS = False
    AttnProcessor2_0 = object
    XFormersAttnProcessor = object

from huggingface_hub import snapshot_download
from torch import amp as torch_amp



# ------------------------------------------------------------
# Small utils
# ------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def from_uint8(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    img = img.convert("RGB").resize(size_hw[::-1], Image.BICUBIC)
    x = torch.from_numpy(np.array(img, dtype=np.float32))
    x = x.permute(2, 0, 1) / 255.0
    x = x * 2.0 - 1.0
    return x


def load_mask(p: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    m = Image.open(p).convert("L").resize(size_hw[::-1], Image.NEAREST)
    t = torch.from_numpy(np.array(m, dtype=np.float32)) / 255.0
    t = (t > 0.5).float().unsqueeze(0)
    return t


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# ---- DDP helpers ----
def maybe_init_distributed() -> Dict[str, int]:
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


def _debug_print_trainables(mod: nn.Module, tag: str, max_items: int = 10):
    total = sum(p.numel() for p in mod.parameters())
    train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    print(f"[trainables/{tag}] total={total/1e6:.2f}M, trainable={train/1e6:.2f}M")

    from collections import Counter
    bucket = Counter()
    for n, p in mod.named_parameters():
        if p.requires_grad:
            head = ".".join(n.split(".")[:3])
            bucket[head] += p.numel()

    top = bucket.most_common(max_items)
    if top:
        print(f"[trainables/{tag}] top{max_items}:", [(k, f"{v/1e6:.2f}M") for k, v in top])


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
            m = load_mask(meta["mask"], (self.H, self.W))
            if self.invert_mask:
                m = 1.0 - m
            x_p_in = x_p * m
        else:
            m = torch.zeros(1, self.H, self.W, dtype=x_p.dtype)
            x_p_in = x_p

        x_concat_in = torch.cat([x_p_in, x_g], dim=2)      # [3,H,2W]
        x_concat_gt = torch.cat([x_p, x_g], dim=2)         # [3,H,2W]
        m_concat = torch.cat([m, torch.ones_like(m)], dim=2)  # [1,H,2W] (right half = 1)

        return {"x_concat_in": x_concat_in, "x_concat_gt": x_concat_gt, "m_concat": m_concat}


# ------------------------------------------------------------
# SD3.5 helpers
# ------------------------------------------------------------
# @torch.no_grad()
# def to_latent_sd3(vae, x_bchw: torch.Tensor) -> torch.Tensor:
#     vdtype = next(vae.parameters()).dtype
#     posterior = vae.encode(x_bchw.to(vdtype)).latent_dist
#     latents = posterior.sample()
#     sf = vae.config.scaling_factor
#     sh = getattr(vae.config, "shift_factor", 0.0)
#     latents = (latents - sh) * sf
#     return latents

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


class ConcatProjector(nn.Module):
    # 16 (z) + 16 (Xi) + 1 (mask) = 33 -> 16
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


# ------------------------------------------------------------
# Freezing & attention backend
# ------------------------------------------------------------
def freeze_all_but_self_attn_qkv(transformer: SD3Transformer2DModel):
    for p in transformer.parameters():
        p.requires_grad = False

    deny_tokens = ("attn2", "cross", "encoder")
    kept = []

    for name, module in transformer.named_modules():
        has_qkv = all(hasattr(module, attr) for attr in ("to_q", "to_k", "to_v", "to_out"))
        if not has_qkv:
            continue
        if any(tok in name for tok in deny_tokens):
            continue  # skip cross-attn

        for subn in ("to_q", "to_k", "to_v", "to_out"):
            subm = getattr(module, subn, None)
            if subm is None:
                continue
            for pn, p in subm.named_parameters(recurse=True):
                p.requires_grad = True
                if len(kept) < 16:
                    kept.append(f"{name}.{subn}.{pn}")

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No params unfrozen; check patterns.")
    return trainable, kept


def apply_attention_backend(transformer: SD3Transformer2DModel, logger: logging.Logger,
                            prefer_xformers: bool = True) -> str:
    try:
        if prefer_xformers:
            transformer.enable_xformers_memory_efficient_attention()
            return "xformers_via_enable()"
    except Exception as e:
        logger.info(f"[mem] xFormers not available via enable(): {e}")
    return "default_attention"


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
class CatVTON_SD3_Trainer:
    def __init__(self, cfg: DotDict, run_dirs: Dict[str, str], cfg_yaml_to_save: Optional[Dict[str, Any]] = None):
        self.cfg = cfg

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

        # ---- Schedulers ----
        self.train_scheduler = FM.from_config(pipe.scheduler.config)
        self.scheduler       = FM.from_config(pipe.scheduler.config)
        # <<< FIX: 스케줄러 prediction_type을 모델 원본 설정에 맞춤
        pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        self.train_scheduler.config.prediction_type = pred_type
        self.scheduler.config.prediction_type       = pred_type
        if self.is_main:
            self.logger.info(f"[sched] training/inference = FlowMatchEuler (pred={pred_type})")
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

        attn_mode = apply_attention_backend(self.transformer, self.logger, prefer_xformers=cfg.prefer_xformers)
        if self.is_main:
            self.logger.info(f"[mem] attention_backend={attn_mode}")

        # projector: keep FP32 — 33ch 입력
        self.projector = ConcatProjector(in_ch=33, out_ch=16).to(self.device, dtype=torch.float32)
        self.adapter_alpha = float(getattr(cfg, "adapter_alpha", 1.0))

        self.start_epoch = 0
        self.start_step = 0
        self._loaded_ckpt_optimizer = None
        self._loaded_ckpt_scaler = None
        self._loaded_rng = None
        self._loaded_cuda_rng = None

        resume_path = cfg.resume_ckpt or cfg.default_resume_ckpt
        self._resume_from_ckpt_if_needed(resume_path)

        # freeze except self-attn Q/K/V/Out (before DDP)
        trainable_tf, keep_names_sample = freeze_all_but_self_attn_qkv(self.transformer)

        # cast trainable(attn) params to FP32 when using fp16 training
        casted = 0
        if self.dtype == torch.float16:
            for _, p in self.transformer.named_parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32); casted += 1
        if self.is_main:
            self.logger.info(f"[dtype] casted_trainable_to_fp32={casted}")

        proj_params = sum(p.numel() for p in self.projector.parameters())
        if self.is_main:
            self.logger.info(f"[freeze] sample_trainable_params (up to 16): {', '.join(keep_names_sample)}")
            print(f"Trainable params (unique): transformer={trainable_tf/1e6:.2f}M, projector={proj_params/1e6:.4f}M")
            _debug_print_trainables(self.transformer, "after_freeze_before_ddp")

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
        self.logger.info("Mask semantics: Mi=1 KEEP, Mi=0 HOLE")

        if self.is_dist:
            self.steps_per_epoch = max(1, self.sampler.num_samples // cfg.batch_size)
        else:
            self.steps_per_epoch = max(1, len(self.dataset) // cfg.batch_size)

        # AMP/GradScaler
        self.use_scaler = (self.device.type == "cuda") and (self.dtype == torch.float16)
        if self.is_main:
            print(f"[amp] dtype={self.dtype}, use_scaler={self.use_scaler}")
        self.logger.info(f"[amp] dtype={self.dtype}, use_scaler={self.use_scaler}")
        self.scaler = None

        # DDP wrap
        self.transformer = DDP(
            self.transformer, device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, find_unused_parameters=False
        )
        self.projector = DDP(
            self.projector, device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, find_unused_parameters=False
        )
        if isinstance(self.projector, DDP):
            self.projector.module.to(dtype=torch.float32)
        else:
            self.projector.to(dtype=torch.float32)

        if self.is_main:
            _debug_print_trainables(self.transformer.module, "after_ddp")

        # ------- optimizer: param groups (transformer vs projector) -------
        tf_params   = [p for p in self.transformer.parameters() if p.requires_grad]
        proj_params = list(self.projector.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": tf_params,   "lr": cfg.lr},
                {"params": proj_params, "lr": cfg.proj_lr},
            ],
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

        if self._loaded_ckpt_optimizer is not None:
            self.optimizer.load_state_dict(self._loaded_ckpt_optimizer)
            dev = next(self.transformer.parameters()).device
            for s in self.optimizer.state.values():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        s[k] = v.to(dev)
            if self.is_main:
                self.logger.info("[resume] optimizer state loaded")

        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
        self._preview_gen = torch.Generator(device=self.device).manual_seed(cfg.preview_seed)
        self._mask_keep_logged = False  # <<< FIX: 마스크 극성 빠른 점검 로그 1회용 플래그

    def _encode_prompts(self, bsz:int):
        if getattr(self.cfg, "disable_text", True):
            # 한 번만 실제 '빈 프롬프트'를 받아 캐시
            if not hasattr(self, "_null_pe"):
                pe, _, ppe, _ = self.encode_prompt(
                    prompt=[""], prompt_2=[""], prompt_3=[""],
                    device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
                )
                self._null_pe  = pe.detach().to(self.dtype)
                self._null_ppe = ppe.detach().to(self.dtype)
            return (self._null_pe.expand(bsz, -1, -1).contiguous(),
                    self._null_ppe.expand(bsz, -1).contiguous())
        # 또는 고정 문장 사용
        text = getattr(self.cfg, "fixed_prompt",
                    "studio photo, front view, clean background, high quality")
        pe, _, ppe, _ = self.encode_prompt([text]*bsz, [text]*bsz, [text]*bsz,
                                        device=self.device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=False)
        return pe.to(self.dtype), ppe.to(self.dtype)

        try:
            pe, _, ppe, _ = self.encode_prompt(
                prompt=empties, prompt_2=empties, prompt_3=empties,
                device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        except TypeError:
            pe, _, ppe, _ = self.encode_prompt(
                prompt=empties,
                device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
        return pe.to(self.dtype), ppe.to(self.dtype)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

    def _fm_scale(self, x: torch.Tensor, sigma) -> torch.Tensor:
        # (호환용): 스케줄러 함수가 없을 때만 사용
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=x.device, dtype=torch.float32)
        sigma = sigma.to(device=x.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma[None]
        s = sigma.view(-1, *([1] * (x.ndim - 1)))
        x_f32 = x.float()
        x_scaled = x_f32 / torch.sqrt(s * s + 1.0)
        return x_scaled.to(x.dtype)
    
    def _ensure_fm_sigmas(self, scheduler, num_steps: int) -> torch.Tensor:
        # <<< FIX: 항상 마지막이 0이 되도록 보장
        scheduler.set_timesteps(num_steps, device=self.device)
        sigmas = getattr(scheduler, "sigmas", None)
        if not (torch.is_tensor(sigmas) and sigmas.numel() > 0):
            sigma_min = float(getattr(scheduler.config, "sigma_min", 0.03))
            sigma_max = float(getattr(scheduler.config, "sigma_max", 14.61))
            rho = float(getattr(scheduler.config, "rho", 7.0))
            i = torch.linspace(0, 1, num_steps, device=self.device, dtype=torch.float32)
            sigmas = (sigma_max ** (1.0 / rho) + i * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
        sigmas = sigmas.to(device=self.device, dtype=torch.float32)
        if sigmas[-1] != 0:
            sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        try:
            scheduler.sigmas = sigmas.clone()
            scheduler.timesteps = sigmas.clone()
        except Exception:
            pass
        return sigmas

    # --- preview helpers ---
    @torch.no_grad()
    def _preview_sample(self, batch, num_steps: int, adapter_alpha: float, global_step: int = 0):
        """
        프리뷰: 오른쪽(KEEP)은 항상 Xi + σ·noise로 유지, 왼쪽(HOLE)만 생성.
        """
        H, W = self.cfg.size_h, self.cfg.size_w
        x_in = batch["x_concat_in"].to(self.device, self.dtype)
        m    = batch["m_concat"].to(self.device, self.dtype)

        # latents & mask
        Xi = to_latent_sd3(self.vae, x_in).to(self.dtype)
        Mi = F.interpolate(m, size=(H // 8, (2 * W) // 8), mode="nearest").to(self.dtype)
        K  = Mi.float()          # KEEP
        Hmask = 1.0 - K
        B = Xi.shape[0]

        # sigmas
        sched_full = FM.from_config(self.scheduler.config)
        sigmas_full = self._ensure_fm_sigmas(sched_full, max(2, int(num_steps)))
        if sigmas_full.numel() < 2:
            return self._denorm(x_in)

        s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
        start_idx = min(int(s * (sigmas_full.numel() - 1)), sigmas_full.numel() - 2)
        sched, sigmas = self._build_preview_scheduler(num_steps, start_idx)

        # float32 noise / state
        noise = torch.randn(Xi.shape, dtype=torch.float32, device=Xi.device, generator=self._preview_gen)
        sigma0 = sigmas[0].to(Xi.device, torch.float32)

        # 시작: KEEP=Xi+σ·noise, HOLE=σ·noise
        z = (K * (Xi.float() + sigma0 * noise) + (1.0 - K) * (sigma0 * noise)).float()

        # prompt
        prompt_embeds, pooled = self._encode_prompts(B)
        model    = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
        proj_net = self.projector.module   if isinstance(self.projector, DDP)   else self.projector

        was_train = self.transformer.training
        self.transformer.eval()
        try:
            for i in range(sigmas.numel() - 1):
                sigma_t   = sigmas[i].to(Xi.device, torch.float32)
                sigma_t_b = sigma_t.expand(B)

                # preconditioning
                try:
                    z_in = sched.scale_model_input(z, sigma_t_b)
                except Exception:
                    z_in = self._fm_scale(z, sigma_t_b)
                z_in_f32 = z_in.float()

                # 33ch concat → projector (fp32) — HOLE만 주입
                hidden_cat = torch.cat([z_in_f32, Xi.float(), Mi.float()], dim=1)
                hidden_cat = torch.nan_to_num(hidden_cat, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    proj = proj_net(hidden_cat).float()
                proj = proj * Hmask

                # norm match
                if self.cfg.norm_match_adapter:
                    eps = 1e-6
                    scale = (z_in_f32.flatten(1).norm(dim=1, keepdim=True) /
                            (proj.flatten(1).norm(dim=1, keepdim=True) + eps)).view(B,1,1,1)
                    proj = proj * scale.clamp(0.0, float(self.cfg.norm_match_clip))

                # 모델 입력
                hidden_in = (z_in_f32 + float(adapter_alpha) * proj).to(self.dtype)

                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=='cuda')):
                    eps_pred = model(
                        hidden_states=hidden_in,
                        timestep=sigma_t_b,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                # Euler step
                z = sched.step(eps_pred, sigma_t, z).prev_sample.float()

                # per-step 재합성: KEEP은 계속 고정
                sigma_next = sigmas[i + 1].to(Xi.device, torch.float32)
                z = (K * (Xi.float() + sigma_next * noise) + (1.0 - K) * z).float()

            # decode & 최종 합성(오른쪽 입력 복사)
            x_hat = from_latent_sd3(self.vae, z.to(self.dtype))
            K3_keep = m.repeat(1, 3, 1, 1)
            x_final = K3_keep * x_in + (1.0 - K3_keep) * x_hat
            return torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
        finally:
            if was_train:
                self.transformer.train()





    def _build_preview_scheduler(self, num_steps: int, start_idx: int):
        steps = max(2, int(num_steps))
        sched = FM.from_config(self.scheduler.config)
        sigmas_full = self._ensure_fm_sigmas(sched, steps)
        start_idx = min(max(0, int(start_idx)), int(sigmas_full.numel()) - 2)
        sigmas_sub = sigmas_full[start_idx:].contiguous()
        # <<< FIX: 잘라낸 후에도 마지막 0 보장
        if sigmas_sub[-1] != 0:
            sigmas_sub = torch.cat([sigmas_sub, sigmas_sub.new_zeros(1)])
        try:
            sched.sigmas = sigmas_sub.clone()
            sched.timesteps = sigmas_sub.clone()
            if hasattr(sched, "step_index"):
                sched.step_index = 0
            if hasattr(sched, "_step_index"):
                sched._step_index = 0
        except Exception:
            pass
        return sched, sigmas_sub
    
    @torch.no_grad()
    def smoke_test_base_sampler(self, batch, num_steps: int = 40):
        """
        샘플러/스케줄러가 정상인지 확인하는 순수 FM Euler 샘플러.
        - projector/concat/재합성 전부 비활성화 (hidden_in = z_in)
        - α=0과 같은 효과
        """
        x_in = batch["x_concat_in"].to(self.device, self.dtype)
        B, _, Hh, Ww2 = x_in.shape
        # latent 해상도 얻기
        Xi = to_latent_sd3(self.vae, x_in).float()
        sched = FM.from_config(self.scheduler.config)
        sigmas = self._ensure_fm_sigmas(sched, max(2, int(num_steps)))

        # 초기화
        noise = torch.randn(Xi.shape, dtype=torch.float32, device=Xi.device, generator=self._preview_gen)
        z = sigmas[0].to(noise) * noise  # float32

        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer

        for i in range(sigmas.numel() - 1):
            sigma_t   = sigmas[i].to(noise)
            sigma_t_b = sigma_t.expand(B)

            try:
                z_in = sched.scale_model_input(z, sigma_t_b)
            except Exception:
                z_in = self._fm_scale(z, sigma_t_b)

            with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=='cuda')):
                eps = model(
                    hidden_states=z_in.to(self.dtype),                # projector 미사용
                    timestep=sigma_t_b,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled,
                    return_dict=True,
                ).sample.float()
            z = sched.step(eps, sigma_t, z).prev_sample.float()

        x = from_latent_sd3(self.vae, z.to(self.dtype))
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0).detach().cpu()

    @torch.no_grad()
    def one_step_teacher_forcing(self, x_in, x_gt, m):
        """
        학습 분포에서 x_t 생성 → ε̂ → z0_hat 복원 → 디코드
        패널 외부에서도 빠르게 확인하고 싶을 때 사용.
        """
        H, W = self.cfg.size_h, self.cfg.size_w
        Xi = to_latent_sd3(self.vae, x_in).float()
        z0 = to_latent_sd3(self.vae, x_gt).float()
        Mi = F.interpolate(m, size=(H//8, (2*W)//8), mode="nearest").float()
        B = Xi.shape[0]
        sigma = self._sample_sigmas(self.train_scheduler, B).view(B,1,1,1).float()
        eps   = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype)        
        x_t   = Mi*(Xi + sigma*eps) + (1.0 - Mi)*(z0 + sigma*eps)

        try:    z_in = self.train_scheduler.scale_model_input(x_t, sigma.view(B))
        except: z_in = self._fm_scale(x_t, sigma.view(B))

        hidden_cat = torch.cat([z_in.float(), Xi.float(), Mi.float()], dim=1)
        hidden_cat = torch.nan_to_num(hidden_cat, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pj = (self.projector.module if isinstance(self.projector, DDP) else self.projector)(hidden_cat).float()
        Hmask = 1.0 - Mi
        pj = pj * Hmask
        hidden_in = (z_in.float() + float(self.adapter_alpha)*pj).to(self.dtype)

        prompt_embeds, pooled = self._encode_prompts(B)
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=='cuda')):
            eps_pred = (self.transformer.module if isinstance(self.transformer, DDP) else self.transformer)(
                hidden_states=hidden_in, timestep=sigma.view(B),
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled, return_dict=True
            ).sample.float()

        z0_hat = x_t - sigma*eps_pred
        return torch.clamp((from_latent_sd3(self.vae, z0_hat.to(self.dtype)) + 1.0)*0.5, 0.0, 1.0).detach().cpu()
    
    @torch.no_grad()
    def infer_tryon_once(
        self,
        x_in: torch.Tensor,   # [B,3,H,2W]
        m: torch.Tensor,      # [B,1,H,2W]  (Mi=1 KEEP, Mi=0 HOLE)
        steps: int = 40,
        alpha: float = 0.8,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        device = self.device
        H, W = self.cfg.size_h, self.cfg.size_w

        # latents / mask
        Xi = to_latent_sd3(self.vae, x_in.to(self.dtype)).float()
        Mi = F.interpolate(m.to(self.dtype), size=(H // 8, (2 * W) // 8), mode="nearest").float()
        K = Mi.float()
        Hmask = 1.0 - K
        B = Xi.shape[0]

        # scheduler/sigmas
        sched = FM.from_config(self.scheduler.config)
        sigmas_full = self._ensure_fm_sigmas(sched, max(2, int(steps)))
        s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
        start_idx = min(int(s * (sigmas_full.numel() - 1)), sigmas_full.numel() - 2)
        sigmas = sigmas_full[start_idx:].contiguous()
        if sigmas[-1] != 0:
            sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        try:
            sched.sigmas = sigmas.clone()
            sched.timesteps = sigmas.clone()
        except Exception:
            pass

        # init
        gen = (torch.Generator(device=device).manual_seed(int(seed))
            if seed is not None else self._preview_gen)
        noise = torch.randn(Xi.shape, dtype=torch.float32, device=Xi.device, generator=gen)
        s0 = sigmas[0].to(Xi)
        z = (K * (Xi + s0 * noise) + (1.0 - K) * (s0 * noise)).float()

        # prompts
        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
        proj_net = self.projector.module if isinstance(self.projector, DDP) else self.projector

        was_train = model.training
        model.eval()
        try:
            for i in range(sigmas.numel() - 1):
                sigma_t = sigmas[i].to(Xi)
                sigma_t_b = sigma_t.expand(B)

                try:
                    z_in = sched.scale_model_input(z, sigma_t_b)
                except Exception:
                    z_in = self._fm_scale(z, sigma_t_b)
                z_in_f32 = z_in.float()

                # 33ch concat → projector (HOLE만)
                hidden_cat = torch.cat([z_in_f32, Xi.float(), Mi.float()], dim=1)
                hidden_cat = torch.nan_to_num(hidden_cat, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    proj = proj_net(hidden_cat).float()
                proj = proj * Hmask

                if self.cfg.norm_match_adapter:
                    eps = 1e-6
                    scale = (z_in_f32.flatten(1).norm(dim=1, keepdim=True) /
                            (proj.flatten(1).norm(dim=1, keepdim=True) + eps)).view(B, 1, 1, 1)
                    proj = proj * scale.clamp(0.0, float(self.cfg.norm_match_clip))

                hidden_in = (z_in_f32 + float(alpha) * proj).to(self.dtype)

                with torch.amp.autocast(device_type="cuda", dtype=self.dtype, enabled=(device.type == "cuda")):
                    eps_pred = model(
                        hidden_states=hidden_in,
                        timestep=sigma_t_b,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                z = sched.step(eps_pred, sigma_t, z).prev_sample.float()

                # 오른쪽 재합성(KEEP 고정)
                sigma_next = sigmas[i + 1].to(Xi)
                z = (K * (Xi + sigma_next * noise) + (1.0 - K) * z).float()

            x_hat = from_latent_sd3(self.vae, z.to(self.dtype))
            K3_keep = m.repeat(1, 3, 1, 1)
            x_final = K3_keep * x_in + (1.0 - K3_keep) * x_hat
            return torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
        finally:
            if was_train:
                model.train()


    @torch.no_grad()
    def _bottom_caption_row(self, names: List[str], widths: List[int],
                            height: int = 28, scale: float = 10.0,
                            bg: float = 1.0, fg: float = 0.0) -> torch.Tensor:
        """
        패널 가장 아래에 한 번만 붙일 전체 폭 캡션 스트립을 생성.
        - names: 각 열 이름
        - widths: 각 열의 pixel 폭(열별 W). sum(widths) == 전체 패널 폭
        - height: 기본 높이
        - scale: 글자/스트립 크기 배수 (10.0 = 10배)
        반환: [1,3,H,W] (0..1, CPU)
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        W_total = int(sum(map(int, widths)))
        H = int(height * scale)

        bg255 = int(bg * 255); fg255 = int(fg * 255)
        img = Image.new("RGB", (W_total, H), (bg255, bg255, bg255))
        draw = ImageDraw.Draw(img)

        # 큰 글꼴 시도(있으면 DejaVu, 없으면 기본 글꼴 + 자동 스케일)
        font = None
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(H * 0.7))
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        x0 = 0
        for name, w in zip(names, widths):
            w = int(w)
            try:
                bbox = draw.textbbox((0, 0), name, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(name, font=font)

            # 각 열 폭의 가운데 정렬
            x = x0 + max(0, (w - tw) // 2)
            y = max(0, (H - th) // 2)
            draw.text((x, y), name, fill=(fg255, fg255, fg255), font=font)
            x0 += w

        t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        return t.unsqueeze(0)  # [1,3,H,W]


    @torch.no_grad()
    def run_infer_once(
        self,
        x_in: torch.Tensor,      # [B,3,H,2W]
        m: torch.Tensor,         # [B,1,H,2W]
        out_path: str,
        steps: int = 60,
        alpha: float = 0.8,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        infer_tryon_once를 호출해 파일로도 저장하는 래퍼.
        """
        img = self.infer_tryon_once(x_in, m, steps=steps, alpha=alpha, seed=seed)
        save_image(make_grid(img, nrow=1), out_path)
        if hasattr(self, "logger"):
            self.logger.info(f"[infer] saved inference to {out_path}")
        return img


    @torch.no_grad()
    def _save_preview(self, batch: Dict[str, torch.Tensor], global_step: int, max_rows: int = 4):
        """
        패널 저장 (xi_noisy_img, pred_img_final, one_step_img 포함)
        - one_step_img: self.one_step_teacher_forcing(x_in, x_gt, m) 호출로 간소화
        """
        if not self.is_main:
            return

        rows = int(self.cfg.get("preview_rows", 1))
        x_in = batch["x_concat_in"][:rows].to(self.device, self.dtype)
        x_gt = batch["x_concat_gt"][:rows].to(self.device, self.dtype)
        m    = batch["m_concat"][:rows].to(self.device, self.dtype)
        if any(t.shape[0] == 0 for t in [x_in, x_gt, m]):
            return

        # 마스크 극성 빠른 점검(1회)
        if not self._mask_keep_logged:
            keep_ratio = (m[:, :, :, : m.shape[-1] // 2].float().mean().item())
            self.logger.info(f"[debug] left-half KEEP ratio ≈ {keep_ratio:.3f} (Mi=1 KEEP)")
            self._mask_keep_logged = True

        # xi_noisy_img (디버그용)
        num_steps = max(2, int(self.cfg.preview_infer_steps))
        Xi_lat = to_latent_sd3(self.vae, x_in).to(self.dtype)
        s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
        _, sigmas_probe = self._build_preview_scheduler(num_steps, min(int(s * (num_steps - 1)), num_steps - 2))
        sigma0 = sigmas_probe[0].to(Xi_lat.device, torch.float32)
        noise0 = torch.randn(Xi_lat.shape, dtype=torch.float32, device=Xi_lat.device, generator=self._preview_gen)
        xi_noisy_img = from_latent_sd3(self.vae, (Xi_lat.float() + sigma0 * noise0).to(self.dtype))
        xi_noisy_img = torch.clamp((xi_noisy_img + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
        
        mini = {
            "x_concat_in": x_in,
            "x_concat_gt": x_gt,
            "m_concat": m,
        }

        alphas: List[float] = list(self.cfg.get("preview_alpha_sweep", [self.adapter_alpha])) or [self.adapter_alpha]
        for a in alphas:
            # ---- (0) 태그를 먼저 준비 ----
            a_tag = f"{float(a):.1f}".replace(".", "_")

            # ---- (1) inpaint 프리뷰 ----
            pred_img_final = self._preview_sample(
                mini, num_steps=num_steps, adapter_alpha=float(a), global_step=global_step
            )

            # ---- (2) 1-step teacher forcing ----
            one_step_img = self.one_step_teacher_forcing(x_in, x_gt, m)

            # ---- (4) 패널 구성 ----
            _, _, Hh, WW = x_gt.shape
            Ww = WW // 2
            B = x_gt.shape[0]

            person        = x_gt[:, :, :, :Ww]
            garment       = x_gt[:, :, :, Ww:]
            mask_keep     = m[:, :, :, :Ww]
            mask_vis      = mask_keep.repeat(1, 3, 1, 1)
            masked_person = person * mask_keep

            def _cpu32(t): return t.detach().to('cpu', dtype=torch.float32, non_blocking=True).contiguous()

            # (4-1) 타일과 라벨 이름 정의
            tiles_and_names = [
                (_cpu32(self._denorm(person))   , "person"),
                (_cpu32(mask_vis)               , "mask_vis"),
                (_cpu32(self._denorm(masked_person)), "masked_person"),
                (_cpu32(self._denorm(garment))  , "garment"),
                (xi_noisy_img                   , "Xi + noise"),
                (_cpu32(self._denorm(x_gt))     , "GT pair"),
                (pred_img_final                 , f"preview α={float(a):.1f}"),
                (one_step_img                   , "1-step TF"),
            ]

            # (4-2) per-tile 캡션 제거: 그냥 열들만 가로 concat
            cols = [t for (t, _) in tiles_and_names]
            panel_bchw = torch.cat(cols, dim=3)                      # [B,3,H, sumW]
            col_widths = [int(t.shape[-1]) for (t, _) in tiles_and_names]
            col_names  = [name for (_, name) in tiles_and_names]

            # (4-3) 배치 전체를 세로로 쌓은 단일 이미지
            grid = make_grid(panel_bchw, nrow=1, padding=0)  # 패딩 제거. [3, B*H, sumW]  ← 한 장

            # (4-4) 맨 아래에 열 이름을 '한 번만' 큰 글씨로 붙임
            caption_h = int(self.cfg.get("preview_caption_h", 28))
            bottom = self._bottom_caption_row(
                names=col_names,
                widths=col_widths,
                height=caption_h,
                scale=2.5,   # 글자 크기
                bg=1.0, fg=0.0
            )[0]                                                    # [3,Hc,sumW]

            final_img = torch.cat([grid, bottom], dim=1)            # 세로 이어붙이기 (H축)

            out_path = os.path.join(self.img_dir, f"step_{global_step:06d}_alpha_{a_tag}.png")
            save_image(final_img, out_path)
            self.logger.info(f"[img] saved preview at step {global_step} (alpha={a}): {out_path}")


    # ------ training core ------

    def _sample_sigmas(self, scheduler, B: int) -> torch.Tensor:
        device = self.device
        has_sigmas = getattr(scheduler, "sigmas", None)
        if torch.is_tensor(has_sigmas) and has_sigmas.numel() > 0:
            sigma_max = float(has_sigmas.max())
            sigma_min = float(has_sigmas.min())
        else:
            sigma_min = float(getattr(scheduler.config, "sigma_min", 0.03))
            sigma_max = float(getattr(scheduler.config, "sigma_max", 14.61))
        u = torch.rand(B, device=device)
        return sigma_min * (sigma_max / sigma_min) ** u

    def _set_phase(self, phase: str):
        assert phase in ("proj_only", "base_only")
        if getattr(self, "_curr_phase", None) == phase:
            return

        if phase == "proj_only":
            # transformer 전부 동결
            tf = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
            for p in tf.parameters():
                p.requires_grad = False
            # projector 학습 on
            pj = self.projector.module if isinstance(self.projector, DDP) else self.projector
            for p in pj.parameters():
                p.requires_grad = True

            # optimizer 재빌드 (projector만)
            pj_params = [p for p in self.projector.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                [{"params": pj_params, "lr": float(self.cfg.get("proj_lr", 1e-3))}],
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
            )

            # sanity check
            tf_train = sum(p.numel() for p in tf.parameters() if p.requires_grad)
            pj_train = sum(p.numel() for p in pj.parameters() if p.requires_grad)
            if tf_train != 0:
                raise RuntimeError(f"[phase=proj_only] transformer still trainable params={tf_train}")
            if self.is_main:
                self.logger.info(f"[phase] switched to proj_only (proj_trainable={pj_train/1e6:.3f}M)")

        else:  # base_only
            pj = self.projector.module if isinstance(self.projector, DDP) else self.projector
            for p in pj.parameters():
                p.requires_grad = False

            tf = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
            for p in tf.parameters():
                p.requires_grad = False
            _ = freeze_all_but_self_attn_qkv(tf)

            tf_params = [p for p in self.transformer.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                [{"params": tf_params, "lr": float(self.cfg.get("lr", 1e-5))}],
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
            )

            tf_train = sum(p.numel() for p in tf.parameters() if p.requires_grad)
            pj_train = sum(p.numel() for p in pj.parameters() if p.requires_grad)
            if pj_train != 0:
                raise RuntimeError(f"[phase=base_only] projector still trainable params={pj_train}")
            if self.is_main:
                self.logger.info(f"[phase] switched to base_only (tf_trainable={tf_train/1e6:.3f}M)")

        self._curr_phase = phase


    def _compute_adapter_alpha(self, global_step: int) -> float:
        max_a = float(self.adapter_alpha)

        if getattr(self, "_curr_phase", None) == "proj_only":
            a0 = float(self.cfg.get("alpha_proj_only_start", 0.3))
            warm = int(self.cfg.get("alpha_proj_only_warmup_steps", 300))
            if warm <= 0:
                return max_a
            t = min(1.0, (global_step + 1) / float(warm))
            return a0 + (max_a - a0) * t

        # base_only
        warm = int(self.cfg.get("alpha_warmup_steps", 2000))
        if warm <= 0:
            return max_a
        t = min(1.0, (global_step + 1) / float(warm))
        return max_a * t



    def step(self, batch, global_step: int, cur_alpha: float,
         cur_cond_dropout_p: float, hole_only: bool):
        H, W = self.cfg.size_h, self.cfg.size_w

        x_concat_in = batch["x_concat_in"].to(self.device, self.dtype)
        x_concat_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m_concat = batch["m_concat"].to(self.device, self.dtype)

        with torch.no_grad():
            Xi = to_latent_sd3(self.vae, x_concat_in).to(self.dtype)   # [B,16,h,w]
            z0 = to_latent_sd3(self.vae, x_concat_gt).to(self.dtype)   # [B,16,h,w]
            Mi = F.interpolate(m_concat, size=(H // 8, (2 * W) // 8), mode="nearest").to(self.dtype)  # [B,1,h,w]

        B = Xi.shape[0]

        # cond-dropout
        if cur_cond_dropout_p > 0:
            drop = (torch.rand(B, device=self.device, dtype=self.dtype) < cur_cond_dropout_p).float().view(B, 1, 1, 1)
            Xi = Xi * (1.0 - drop)
            Mi = Mi * (1.0 - drop)

        # K=KEEP(=Mi), Hmask=HOLE
        K = Mi.float()
        Hmask = 1.0 - K

        # sample sigma
        sigmas = self._sample_sigmas(self.train_scheduler, B)

        # inpainting forward (KEEP=Xi, HOLE=z0)
        z0_f32     = z0.float()
        Xi_f32     = Xi.float()
        noise_f32  = torch.randn(z0_f32.shape, device=z0_f32.device, dtype=z0_f32.dtype)  # ε
        sigma_b11  = sigmas.view(B, 1, 1, 1).to(dtype=torch.float32)
        z0_noisy   = z0_f32 + sigma_b11 * noise_f32
        Xi_noisy   = Xi_f32 + sigma_b11 * noise_f32
        x_t_mixed  = K * Xi_noisy + Hmask * z0_noisy

        target_f32 = noise_f32  # ε-target

        # preconditioning
        try:
            z_in = self.train_scheduler.scale_model_input(x_t_mixed, sigmas)
        except Exception:
            z_in = self._fm_scale(x_t_mixed, sigmas)

        # 33ch concat → projector
        hidden_cat_f32 = torch.cat([z_in.float(), Xi.float(), Mi.float()], dim=1)  # [B,33,h,w]
        hidden_cat_f32 = torch.nan_to_num(hidden_cat_f32, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30.0, 30.0)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            proj_f32 = (self.projector.module(hidden_cat_f32.float()) if isinstance(self.projector, DDP)
                        else self.projector(hidden_cat_f32.float()))  # [B,16,h,w]
        proj_f32 = proj_f32.float()

        if self.cfg.norm_match_adapter:
            eps = 1e-6
            s = (z_in.float().flatten(1).norm(dim=1, keepdim=True) /
                (proj_f32.flatten(1).norm(dim=1, keepdim=True) + eps)).view(B,1,1,1)
            proj_f32 = proj_f32 * s.clamp(0.0, float(self.cfg.norm_match_clip))

        # projector는 항상 HOLE만 주입
        proj_f32 = proj_f32 * Hmask

        hidden_in = (z_in.float() + float(cur_alpha) * proj_f32).to(self.dtype)

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(B)

        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
            model = self.transformer
            eps_pred = model(
                hidden_states=hidden_in,
                timestep=sigmas,    # FlowMatch uses sigmas as "t"
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            ).sample

        eps_pred_f32 = eps_pred.float()

        # --------- Base losses ---------
        # ε-MSE: HOLE만
        if bool(self.cfg.get("loss_sigma_weight", False)):
            sigma_w = 1.0 / (sigma_b11 ** 2 + 1.0)
            loss_eps = torch.mean(((eps_pred_f32 - target_f32) ** 2) * Hmask * sigma_w)
        else:
            loss_eps = torch.mean(((eps_pred_f32 - target_f32) ** 2) * Hmask)

        # latent x0 재구성: HOLE만
        z0_hat = x_t_mixed.float() - sigma_b11 * eps_pred_f32
        loss_lat = ((z0_hat - z0_f32) ** 2 * Hmask).sum() / Hmask.sum().clamp_min(1.0)
        rec_lambda = float(self.cfg.latent_rec_lambda)
        warmup_steps = int(getattr(self.cfg, "latent_rec_lambda_warmup_steps", 0))
        if warmup_steps > 0 and global_step < warmup_steps:
            rec_lambda = 0.0

        # KEEP consistency (옵션)
        keep_lambda = float(self.cfg.get("keep_consistency_lambda", 0.0))
        loss_keep = torch.tensor(0.0, device=self.device)
        if keep_lambda > 0.0:
            loss_keep = torch.mean(((z0_hat - z0_f32) ** 2) * K)

        # Garment recon (옵션)
        gar_lambda = float(self.cfg.get("garment_rec_lambda", 0.0))
        if gar_lambda > 0.0:
            Wlat = Mi.shape[-1]
            Rmask = torch.zeros_like(Mi); Rmask[..., :, :, Wlat//2:] = 1.0
            loss_gar = (((z0_hat - z0_f32) ** 2) * Rmask).sum() / Rmask.sum().clamp_min(1.0)
        else:
            loss_gar = torch.tensor(0.0, device=self.device)

        # --------- Projector-specific regularizers ---------
        # (1) 출력 L2 에너지(옵션)
        proj_out_l2_lambda = float(self.cfg.get("proj_out_l2_lambda", 0.0))
        loss_proj_l2 = torch.tensor(0.0, device=self.device)
        if proj_out_l2_lambda > 0.0:
            loss_proj_l2 = (proj_f32.pow(2) * Hmask).mean()

        # (2) 마진 향상(옵션, transformer 1회 추가 추론)
        proj_margin_lambda = float(self.cfg.get("proj_margin_lambda", 0.0))
        proj_margin = float(self.cfg.get("proj_margin", 0.0))
        loss_margin = torch.tensor(0.0, device=self.device)
        if proj_margin_lambda > 0.0:
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
                eps_base = model(
                    hidden_states=z_in.to(self.dtype),   # projector 미적용 경로
                    timestep=sigmas,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=True,
                ).sample.float().detach()  # 베이스는 detatch
            mse_proj = ((eps_pred_f32 - target_f32) ** 2 * Hmask).mean()
            mse_base = ((eps_base      - target_f32) ** 2 * Hmask).mean()
            loss_margin = torch.clamp(mse_proj - mse_base + proj_margin, min=0.0)

        # --------- Total ---------
        loss_total = (
            loss_eps
            + rec_lambda * loss_lat
            + gar_lambda * loss_gar
            + keep_lambda * loss_keep
            + proj_out_l2_lambda * loss_proj_l2
            + proj_margin_lambda * loss_margin
        )

        if not torch.isfinite(loss_total) or not loss_total.requires_grad:
            return None

        # 새 항들도 반환해서 로그 가능하게
        return (
            loss_total,
            float(loss_eps.detach().cpu()),
            float(loss_lat.detach().cpu()),
            float(loss_keep.detach().cpu()),
            float(loss_proj_l2.detach().cpu()),
            float(loss_margin.detach().cpu()),
        )




    def _save_ckpt(self, epoch: int, train_loss_epoch: float, global_step: int) -> str:
        ckpt_path = os.path.join(self.model_dir, f"epoch_{epoch}_loss_{train_loss_epoch:.04f}.ckpt")
        payload = {
            "transformer": ddp_state_dict(self.transformer),
            "projector": ddp_state_dict(self.projector),
            "optimizer": self.optimizer.state_dict(),
            "cfg": dict(self.cfg),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_epoch": float(train_loss_epoch),
            "scaler_state": (self.scaler.state_dict() if (self.scaler is not None and self.use_scaler) else None),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "scheduler_config": (
                self.scheduler.config.to_diff_dict() if hasattr(self.scheduler.config, "to_diff_dict")
                else dict(self.scheduler.config)
            ),
        }
        if self.is_main:
            torch.save(payload, ckpt_path)
            self.logger.info(f"[save] {ckpt_path}")
        return ckpt_path

    def _resume_from_ckpt_if_needed(self, resume_path: Optional[str]):
        self.start_epoch = 0
        self.start_step = 0
        self._loaded_ckpt_optimizer = None

        if not resume_path or not os.path.isfile(resume_path):
            return

        ckpt = torch.load(resume_path, map_location="cpu")

        try:
            self.transformer.load_state_dict(ckpt["transformer"], strict=False)
        except Exception:
            self.transformer.load_state_dict(ckpt["transformer"], strict=False)

        try:
            self.projector.load_state_dict(ckpt["projector"], strict=False)
        except Exception:
            self.projector.load_state_dict(ckpt["projector"], strict=False)

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.start_step  = int(ckpt.get("global_step", 0))

        self._loaded_ckpt_optimizer = ckpt.get("optimizer", None)
        self._loaded_ckpt_scaler = ckpt.get("scaler_state", None)
        self._loaded_rng = ckpt.get("rng_state", None)
        self._loaded_cuda_rng = ckpt.get("cuda_rng_state_all", None)

        if self.is_main:
            self.logger.info(f"[resume] loaded {resume_path} | start_epoch={self.start_epoch}, start_step={self.start_step}")
            print(f"[resume] loaded {resume_path} | start_epoch={self.start_epoch}, start_step={self.start_step}")

    def _grad_norm_sum(self, module: nn.Module) -> float:
        s = 0.0
        for p in module.parameters():
            if p.requires_grad and (p.grad is not None):
                try:
                    s += float(p.grad.detach().float().norm().cpu())
                except Exception:
                    pass
        return s

    def train(self):
        global_step = getattr(self, "start_step", 0)
        epoch = getattr(self, "start_epoch", 0)
        self.transformer.train()
        self.projector.train()
        self.scaler = torch_amp.GradScaler(enabled=self.use_scaler)

        if self._loaded_ckpt_scaler is not None and self.use_scaler:
            self.scaler.load_state_dict(self._loaded_ckpt_scaler)
        if self._loaded_rng is not None:
            torch.set_rng_state(self._loaded_rng)
        if self._loaded_cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self._loaded_cuda_rng)

        # 단계1(projector만) → 단계2(base만)
        proj_first_steps = int(getattr(self.cfg, "proj_first_steps", 0))
        if proj_first_steps > 0 and global_step < proj_first_steps:
            self._set_phase("proj_only")
        else:
            self._set_phase("base_only")

        data_iter = itertools.cycle(self.loader)

        pbar = tqdm(
            total=self.cfg.max_steps,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}",
            leave=True,
            disable=(not self.is_main) or (not sys.stdout.isatty()),
            initial=global_step,
        )

        if self.is_dist and self.sampler is not None:
            self.sampler.set_epoch(epoch)

        # rolling meters
        comp_eps_sum = 0.0
        comp_lat_sum = 0.0
        comp_keep_sum = 0.0
        comp_proj_l2_sum = 0.0
        comp_margin_sum = 0.0

        while global_step < self.cfg.max_steps:
            if proj_first_steps > 0 and global_step == proj_first_steps:
                self._set_phase("base_only")

            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            comp_eps_accum = 0.0
            comp_lat_accum = 0.0
            comp_keep_accum = 0.0
            comp_proj_l2_accum = 0.0
            comp_margin_accum = 0.0
            nonfinite = False
            reason = ""

            # ----- schedules -----
            cur_alpha = self._compute_adapter_alpha(global_step)

            if global_step < int(self.cfg.cond_dropout_warmup_steps):
                cur_cond_dropout_p = 0.0
            else:
                cur_cond_dropout_p = float(self.cfg.cond_dropout_p)

            hole_only = (global_step < int(self.cfg.hole_only_warmup_steps))

            for _ in range(self.cfg.grad_accum):
                batch = next(data_iter)
                out = self.step(batch, global_step, cur_alpha, cur_cond_dropout_p, hole_only)

                if (out is None):
                    nonfinite = True; reason = "None/NaN in forward"; break

                (loss, loss_eps_val, loss_lat_val, loss_keep_val,
                loss_proj_l2_val, loss_margin_val) = out

                if (not torch.isfinite(loss)) or (not loss.requires_grad):
                    nonfinite = True
                    reason = "non-finite/detached loss"; break

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += float(loss.detach().cpu())
                comp_eps_accum += float(loss_eps_val)
                comp_lat_accum += float(loss_lat_val)
                comp_keep_accum += float(loss_keep_val)
                comp_proj_l2_accum += float(loss_proj_l2_val)
                comp_margin_accum += float(loss_margin_val)

            if nonfinite:
                msg = f"[warn] skipping step {global_step}: {reason}."
                if self.is_main:
                    pbar.write(msg)
                self.logger.info(msg)
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_scaler:
                    self.scaler.update()
                global_step += 1
                if self.is_main:
                    pbar.update(1)
                continue

            # grad norms (before unscale/clip/step)
            tf_grad_norm = self._grad_norm_sum(self.transformer)
            pj_grad_norm = self._grad_norm_sum(self.projector)

            do_clip = True
            if self.use_scaler:
                try:
                    self.scaler.unscale_(self.optimizer)
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
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            global_step += 1
            if self.is_main:
                pbar.update(1)

            # update rolling meters
            self._epoch_loss_sum += loss_accum / max(1, self.cfg.grad_accum)
            self._epoch_loss_count += 1
            comp_eps_sum += comp_eps_accum / max(1, self.cfg.grad_accum)
            comp_lat_sum += comp_lat_accum / max(1, self.cfg.grad_accum)
            comp_keep_sum += comp_keep_accum / max(1, self.cfg.grad_accum)
            comp_proj_l2_sum += comp_proj_l2_accum / max(1, self.cfg.grad_accum)
            comp_margin_sum += comp_margin_accum / max(1, self.cfg.grad_accum)

            train_loss_avg = self._epoch_loss_sum / max(1, self._epoch_loss_count)
            loss_eps_avg   = comp_eps_sum / max(1, self._epoch_loss_count)
            loss_lat_avg   = comp_lat_sum / max(1, self._epoch_loss_count)
            loss_keep_avg  = comp_keep_sum / max(1, self._epoch_loss_count)
            loss_l2_avg    = comp_proj_l2_sum / max(1, self._epoch_loss_count)
            loss_margin_avg= comp_margin_sum / max(1, self._epoch_loss_count)

            if self.is_main and ((global_step % self.cfg.log_every) == 0 or global_step == 1):
                self.tb.add_scalar("train/loss_total", train_loss_avg, global_step)
                self.tb.add_scalar("train/loss_eps",   loss_eps_avg,   global_step)
                self.tb.add_scalar("train/loss_lat",   loss_lat_avg,   global_step)
                self.tb.add_scalar("train/loss_keep",  loss_keep_avg,  global_step)
                self.tb.add_scalar("train/loss_proj_l2", loss_l2_avg, global_step)
                self.tb.add_scalar("train/loss_proj_margin", loss_margin_avg, global_step)
                self.tb.add_scalar("train/alpha", cur_alpha, global_step)
                self.tb.add_scalar("train/cond_dropout_p", cur_cond_dropout_p, global_step)
                self.tb.add_scalar("train/hole_only", float(hole_only), global_step)
                self.tb.add_scalar("train/phase_proj_only", float(self._curr_phase == "proj_only"), global_step)
                self.tb.add_scalar("train/grad_norm_transformer", tf_grad_norm, global_step)
                self.tb.add_scalar("train/grad_norm_projector",  pj_grad_norm,  global_step)
                prog = (global_step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0.0
                pct = int(prog * 100)
                line = (f"Epoch {epoch}: {pct:3d}% | step {global_step}/{self.cfg.max_steps} "
                        f"| total={train_loss_avg:.4f} | eps={loss_eps_avg:.4f} | lat={loss_lat_avg:.4f} | keep={loss_keep_avg:.4f} "
                        f"| l2={loss_l2_avg:.5f} | mar={loss_margin_avg:.5f} "
                        f"| α={cur_alpha:.3f} | cd={cur_cond_dropout_p:.2f} | holeOnly={hole_only} | phase={self._curr_phase}")
                pbar.set_postfix_str(f"tot={train_loss_avg:.4f}")
                pbar.write(line)
                self.logger.info(line)

            if self.is_main and ((global_step % self.cfg.image_every) == 0 or global_step == 1):
                try:
                    batch_vis = next(data_iter)
                    self._save_preview(batch_vis, global_step, max_rows=min(4, self.cfg.batch_size))
                    pbar.write(f"[img] saved preview at step {global_step}")
                except Exception as e:
                    msg = f"[warn] preview save failed: {e}"
                    pbar.write(msg); self.logger.info(msg)

            if (global_step % self.steps_per_epoch) == 0:
                epoch += 1
                if self.is_main and (epoch % self.cfg.save_epoch_ckpt) == 0:
                    path = self._save_ckpt(epoch, train_loss_avg, global_step)
                    pbar.write(f"[save-epoch] {path}")
                self._epoch_loss_sum = 0.0
                self._epoch_loss_count = 0
                comp_eps_sum = 0.0
                comp_lat_sum = 0.0
                comp_keep_sum = 0.0
                comp_proj_l2_sum = 0.0
                comp_margin_sum = 0.0
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
    "list_file": None,
    "sd3_model": "stabilityai/stable-diffusion-3.5-large",
    "size_h": 512, "size_w": 384,
    "mask_based": True, "invert_mask": False,

    "lr": 1e-5,          # transformer
    "proj_lr": 3e-4,     # projector
    "batch_size": 4, "grad_accum": 1, "max_steps": 128000,
    "seed": 1337, "num_workers": 4,
    "cond_dropout_p": 0.1,
    "cond_dropout_warmup_steps": 2000,
    "alpha_warmup_steps": 2000,
    "hole_only_warmup_steps": 2000,
    "mixed_precision": "fp16",
    "loss_sigma_weight": False,
    "adapter_alpha": 1.0,
    "use_scaler": True,

    "prefer_xformers": True,
    "strip_cross_attention": True,
    "disable_text": True,

    "save_root_dir": "logs", "save_name": "catvton_sd35",
    "log_every": 50, "image_every": 500, "save_every": 12800,
    "save_epoch_ckpt": 15,
    "default_resume_ckpt": None,
    "resume_ckpt": None,

    "preview_infer_steps": 16,
    "preview_strength": 0.6,
    # <<< FIX: 프리뷰 스윕에 0.0 포함
    "preview_alpha_sweep": [0.0, 0.5, 1.0],
    "preview_use_inpaint_start": True,
    "preview_offload_cpu" : True,
    "preview_rows" : 4,
    "preview_save_raw": False, 

    "hole_loss_weight": 2.0,
    "norm_match_adapter": True,
    "norm_match_clip": 5.0,

    "keep_consistency_lambda": 0.0,
    "proj_hole_gate_warmup_steps": 4000,

    "latent_rec_lambda": 0.75,
    "latent_rec_lambda_warmup_steps": 4000,
    "preview_save_base": True,

    "hf_token": None,
}

def parse_float_list_csv(s: str) -> List[float]:
    parts = re.split(r"[,\s]+", s.strip())
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    return vals

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/catvton_sd35.yaml",
                   help="YAML config path. CLI overrides YAML.")
    p.add_argument("--list_file", type=str, default=None)
    p.add_argument("--sd3_model", type=str, default=None)
    p.add_argument("--size_h", type=int, default=None)
    p.add_argument("--size_w", type=int, default=None)
    p.add_argument("--mask_based", action="store_true")
    p.add_argument("--mask_free", action="store_true")
    p.add_argument("--invert_mask", action="store_true")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--proj_lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--image_every", type=int, default=None)
    p.add_argument("--cond_dropout_p", type=float, default=None)
    p.add_argument("--cond_dropout_warmup_steps", type=int, default=None)
    p.add_argument("--alpha_warmup_steps", type=int, default=None)
    p.add_argument("--hole_only_warmup_steps", type=int, default=None)
    p.add_argument("--mixed_precision", type=str, default=None, choices=["fp16", "fp32", "bf16"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    p.add_argument("--prefer_xformers", action="store_true")
    p.add_argument("--no_prefer_xformers", action="store_true")
    p.add_argument("--strip_cross_attention", action="store_true")
    p.add_argument("--keep_cross_attention", action="store_true")
    p.add_argument("--disable_text", action="store_true")
    p.add_argument("--enable_text", action="store_true")

    p.add_argument("--save_root_dir", type=str, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--preview_infer_steps", type=int, default=None)
    p.add_argument("--preview_strength", type=float, default=None)
    p.add_argument("--preview_seed", type=int, default=None)
    p.add_argument("--preview_alpha_sweep", type=str, default=None,
                   help="Comma/space separated list, e.g. '0.5,1.0,2.0'")
    p.add_argument("--resume_ckpt", type=str, default=None, help="path to checkpoint to resume from")

    # kept for compat
    p.add_argument("--hole_loss_weight", type=float, default=None)
    p.add_argument("--no_norm_match_adapter", action="store_true")
    p.add_argument("--norm_match_clip", type=float, default=None)

    # new
    p.add_argument("--keep_consistency_lambda", type=float, default=None)
    p.add_argument("--latent_rec_lambda", type=float, default=None)
    p.add_argument("--proj_hole_gate_warmup_steps", type=int, default=None)
    p.add_argument("--preview_use_inpaint_start", action="store_true")

    return p.parse_args()


def load_merge_config(args: argparse.Namespace) -> DotDict:
    cfg = dict(DEFAULTS)

    if args.config and os.path.isfile(args.config):
        if yaml is None:
            raise RuntimeError("PyYAML not installed. `pip install pyyaml` or omit --config.")
        with open(args.config, "r") as f:
            y = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in y.items() if v is not None})

    for k in list(cfg.keys()):
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None and not isinstance(v, bool):
                cfg[k] = v

    if getattr(args, "mask_free", False):
        cfg["mask_based"] = False
    elif getattr(args, "mask_based", False):
        cfg["mask_based"] = True

    if getattr(args, "invert_mask", False):
        cfg["invert_mask"] = True

    if getattr(args, "no_prefer_xformers", False):
        cfg["prefer_xformers"] = False
    elif getattr(args, "prefer_xformers", False):
        cfg["prefer_xformers"] = True

    if getattr(args, "keep_cross_attention", False):
        cfg["strip_cross_attention"] = False
    elif getattr(args, "strip_cross_attention", False):
        cfg["strip_cross_attention"] = True

    if getattr(args, "enable_text", False):
        cfg["disable_text"] = False
    elif getattr(args, "disable_text", False):
        cfg["disable_text"] = True

    if getattr(args, "no_norm_match_adapter", False):
        cfg["norm_match_adapter"] = False

    if isinstance(cfg.get("preview_alpha_sweep"), str):
        cfg["preview_alpha_sweep"] = parse_float_list_csv(cfg["preview_alpha_sweep"])
    elif args.preview_alpha_sweep is not None:
        cfg["preview_alpha_sweep"] = parse_float_list_csv(args.preview_alpha_sweep)

    if getattr(args, "proj_hole_gate_warmup_steps", None) is not None:
        cfg["proj_hole_gate_warmup_steps"] = int(args.proj_hole_gate_warmup_steps)

    if getattr(args, "preview_use_inpaint_start", False):
        cfg["preview_use_inpaint_start"] = True

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

    dinfo = maybe_init_distributed()
    is_dist = dinfo["is_dist"]
    rank = dinfo["rank"]

    run_name = None
    if rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_{cfg.save_name}"
    run_name = bcast_object(run_name, src=0)
    run_dirs = build_run_dirs(cfg, run_name=run_name, create=(rank == 0))

    cfg_yaml_to_save = dict(cfg) if rank == 0 else None

    trainer = CatVTON_SD3_Trainer(cfg, run_dirs, cfg_yaml_to_save=cfg_yaml_to_save)
    trainer.train()

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
