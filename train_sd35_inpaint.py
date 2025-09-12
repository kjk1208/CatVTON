# ddp_train_stage1.py — SD3.5 CatVTON (DDP-ready, true inpainting; NO projector)
# - Train: FlowMatch (ε-target) + HOLE-only loss + (optional) masked latent x0 recon
# - Infer/Preview: FlowMatch Euler + per-step recomposition (official inpaint style)
# - Text: unconditional embeds (cross-attn 사실상 무력화; AdaLN 게이팅 0 문제 없음)
# - CatVTON 입출력 유지: person||cloth (W축 concat), person mask 사용(Mi=1 KEEP)
# - Freeze: self-attn Q/K/V/Out만 학습

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
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60)
        )
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
        x_concat_gt = torch.cat([x_p,   x_g], dim=2)       # [3,H,2W]
        m_concat    = torch.cat([m, torch.ones_like(m)], dim=2)  # [1,H,2W] (right half = 1)

        return {"x_concat_in": x_concat_in, "x_concat_gt": x_concat_gt, "m_concat": m_concat}


# ------------------------------------------------------------
# SD3.5 helpers
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Freezing & attention backend
# ------------------------------------------------------------
def freeze_all_but_self_attn_qkv(transformer, open_to_add_out: bool=False, open_io: bool=True):
    # 0) 모두 동결
    for p in transformer.parameters():
        p.requires_grad = False

    # 1) self-attn Q/K/V/Out만 해제
    for name, m in transformer.named_modules():
        if name.endswith(".attn") or hasattr(m, "to_q"):
            for attr in ["to_q","to_k","to_v","to_out"]:
                sub = getattr(m, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad = True

    # 2) cross 경로: 기본은 전부 동결, 단 to_add_out만 옵션으로 오픈
    for name, m in transformer.named_modules():
        if name.endswith(".attn"):
            # 항상 동결
            for attr in ["add_k_proj","add_v_proj","add_q_proj"]:
                sub = getattr(m, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad = False
            # 옵션: to_add_out만 열기
            sub = getattr(m, "to_add_out", None)
            if sub is not None:
                for p in sub.parameters():
                    p.requires_grad = bool(open_to_add_out)

    # 3) 입/출력 적응(패치임베더/헤드)도 열기(권장)
    if open_io:
        if hasattr(transformer, "pos_embed") and hasattr(transformer.pos_embed, "proj"):
            for p in transformer.pos_embed.proj.parameters():
                p.requires_grad = True
        for p in transformer.norm_out.parameters():
            p.requires_grad = True
        for p in transformer.proj_out.parameters():
            p.requires_grad = True

    # 통계
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    kept = [n for n, p in transformer.named_parameters() if p.requires_grad][:16]
    if trainable == 0:
        raise RuntimeError("No params unfrozen; check patterns.")

    # 안전 확인: add_*는 to_add_out 외엔 모두 동결
    for n, p in transformer.named_parameters():
        if (".attn.add_" in n) and (".attn.to_add_out" not in n) and p.requires_grad:
            raise RuntimeError(f"Cross add path unexpectedly trainable: {n}")

    return trainable, kept


def sanity_check_self_vs_cross(model: nn.Module, tag: str = ""):
    self_cnt = 0
    cross_cnt = 0
    for _, m in model.named_modules():
        is_cross = getattr(m, "is_cross_attention", None)
        if is_cross is None:
            is_cross = bool(getattr(m, "cross_attention_dim", 0))

        if any(hasattr(m, a) for a in ("to_q","q_proj","qkv","qkv_proj","to_k","k_proj","to_v","v_proj","to_out")):
            for p in m.parameters():
                if p.requires_grad:
                    if is_cross: cross_cnt += p.numel()
                    else:        self_cnt  += p.numel()

    print(f"[sanity{(':'+tag) if tag else ''}] self-attn trainable={self_cnt/1e6:.2f}M, cross-attn trainable={cross_cnt/1e6:.2f}M")


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

        # HF token
        env_token = (os.environ.get("HUGGINGFACE_TOKEN")
                     or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                     or os.environ.get("HF_TOKEN"))
        token = cfg.hf_token or env_token
        if token is None and self.is_main:
            msg = "[warn] No HF token found. If the repo is gated, set HUGGINGFACE_TOKEN or pass --hf_token."
            print(msg); self.logger.info(msg)

        # Load SD3.5 weights (rank0 download, others cache)
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
        pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        self.train_scheduler.config.prediction_type = pred_type
        self.scheduler.config.prediction_type       = pred_type
        if self.is_main:
            self.logger.info(f"[sched] training/inference = FlowMatchEuler (pred={pred_type})")
            self.logger.info(f"Loaded SD3 model: {cfg.sd3_model}")

        # memory knobs
        try:
            self.transformer.enable_gradient_checkpointing()
            if self.is_main: print("[mem] gradient checkpointing ON")
        except Exception as e:
            if self.is_main: print(f"[mem] gradient checkpointing not available: {e}")

        # Freeze except self-attn Q/K/V/Out
        trainable_tf, keep_names_sample = freeze_all_but_self_attn_qkv(self.transformer)
        if self.is_main:
            sanity_check_self_vs_cross(self.transformer, tag="init_before_ddp")
        if self.dtype == torch.float16:
            for _, p in self.transformer.named_parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)

        if self.is_main:
            print(f"Trainable params (transformer only)={trainable_tf/1e6:.2f}M")
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

        # AMP
        self.use_scaler = (self.device.type == "cuda") and (self.dtype == torch.float16)
        if self.is_main:
            print(f"[amp] dtype={self.dtype}, use_scaler={self.use_scaler}")

        # DDP wrap
        self.transformer = DDP(
            self.transformer, device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, find_unused_parameters=False
        )
        if self.is_main:
            sanity_check_self_vs_cross(self.transformer.module, tag="init_after_ddp")
            _debug_print_trainables(self.transformer.module, "after_ddp")

        # optimizer: 파라미터 그룹 (QKV/OUT, IO, to_add_out)
        qkv_params, add_out_params, io_params = [], [], []
        for n, p in self.transformer.named_parameters():
            if not p.requires_grad:
                continue
            # self-attn 경로
            if (".attn.to_q" in n) or (".attn.to_k" in n) or (".attn.to_v" in n) or (".attn.to_out" in n):
                qkv_params.append(p)
            # cross 추가 출력 경로 (freeze 함수에서 open_to_add_out=True일 때만 모임)
            elif ".attn.to_add_out" in n:
                add_out_params.append(p)
            # 입출력 적응 레이어
            elif ("pos_embed.proj" in n) or ("norm_out" in n) or ("proj_out" in n):
                io_params.append(p)

        param_groups = []
        if qkv_params:
            param_groups.append({"params": qkv_params, "lr": cfg.lr})                # 예: 5e-5
        if io_params:
            param_groups.append({"params": io_params, "lr": cfg.lr * 2.0})          # 예: 1e-4
        if add_out_params:
            param_groups.append({"params": add_out_params, "lr": cfg.lr * 0.5})     # 예: 2.5e-5

        # 혹시라도 아무 그룹도 잡히지 않으면(예: 전부 동결) 안전한 fallback
        if not param_groups:
            tf_params = [p for p in self.transformer.parameters() if p.requires_grad]
            param_groups = [{"params": tf_params, "lr": cfg.lr}]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

        # resume (if any)
        self.start_epoch = 0
        self.start_step  = 0
        resume_path = cfg.resume_ckpt or cfg.default_resume_ckpt
        self._resume_from_ckpt_if_needed(resume_path)

        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
        self._preview_gen = torch.Generator(device=self.device).manual_seed(cfg.preview_seed)
        self._mask_keep_logged = False

        # scaler
        self.scaler = torch_amp.GradScaler(enabled=self.use_scaler)
        self._dumped_mask_once = False
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # ---------------- utilities ----------------
    def _encode_prompts(self, bsz:int):
        # '빈 프롬프트'의 unconditional 임베딩을 사용 => cross-attn 영향 최소화(게이팅 0 아님)
        if not hasattr(self, "_null_pe"):
            pe, _, ppe, _ = self.encode_prompt(
                prompt=[""], prompt_2=[""], prompt_3=[""],
                device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
            self._null_pe  = pe.detach().to(self.dtype)
            self._null_ppe = ppe.detach().to(self.dtype)
        return (self._null_pe.expand(bsz, -1, -1).contiguous(),
                self._null_ppe.expand(bsz, -1).contiguous())

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

    def _fm_scale(self, x: torch.Tensor, sigma) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=x.device, dtype=torch.float32)
        sigma = sigma.to(device=x.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma[None]
        s = sigma.view(-1, *([1] * (x.ndim - 1)))
        x_f32 = x.float()
        x_scaled = x_f32 / torch.sqrt(s * s + 1.0)
        return x_scaled.to(x.dtype)

    def _set_fm_timesteps(self, scheduler, num_steps: int) -> torch.Tensor:
        H, W = self.cfg.size_h, self.cfg.size_w
        sched = FM.from_config(scheduler.config)
        kwargs = {}
        # SD3는 latent 채널=16, patch_size=2(모델 설정 참조)
        tf = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
        patch = getattr(tf, "config").patch_size
        vae_sf = self.vae_scale_factor

        if getattr(sched.config, "use_dynamic_shifting", False):
            image_seq_len = (H // vae_sf // patch) * ((2 * W) // vae_sf // patch)  # 너는 W축 concat이라 2*W
            base_len  = getattr(sched.config, "base_image_seq_len", 256)
            max_len   = getattr(sched.config, "max_image_seq_len", 4096)
            base_s    = getattr(sched.config, "base_shift", 0.5)
            max_s     = getattr(sched.config, "max_shift", 1.16)
            mu = base_s + (max_s - base_s) * (image_seq_len - base_len) / max(1, (max_len - base_len))
            kwargs["mu"] = float(mu)

        sched.set_timesteps(num_steps, device=self.device, **kwargs)
        # FM에선 timesteps=σ 벡터; 끝이 0이 아니면 0 추가
        sigmas = sched.timesteps
        if sigmas[-1] != 0:
            sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
            sched.timesteps = sigmas
        sched.sigmas = sigmas  # 추가 권장
        return sched, sigmas

    def _ensure_fm_sigmas(self, scheduler, num_steps: int) -> torch.Tensor:
        """이제는 _set_fm_timesteps를 통해 스케줄러가 직접 σ를 만들도록 위임한다(mus 포함)."""
        H, W = self.cfg.size_h, self.cfg.size_w
        sched, sigmas = self._set_fm_timesteps(scheduler, num_steps)
        # 스케줄러 객체도 최신 timesteps로 업데이트
        try:
            scheduler.sigmas    = sigmas.clone()
            scheduler.timesteps = sigmas.clone()
        except Exception:
            pass
        return sched, sigmas

    # ---------------- preview / inference ----------------
    @torch.no_grad()
    def _build_preview_scheduler(self, num_steps: int, start_idx: int):
        steps = max(2, int(num_steps))
        # mu 반영된 전체 스케줄 생성
        H, W = self.cfg.size_h, self.cfg.size_w
        sched_full, sigmas_full = self._set_fm_timesteps(self.scheduler, steps)

        start_idx = min(max(0, int(start_idx)), int(sigmas_full.numel()) - 2)
        sigmas_sub = sigmas_full[start_idx:].contiguous()
        if sigmas_sub[-1] != 0:
            sigmas_sub = torch.cat([sigmas_sub, sigmas_sub.new_zeros(1)])

        # 서브 스케줄러 생성
        sched = FM.from_config(self.scheduler.config)
        try:
            sched.sigmas    = sigmas_sub.clone()
            sched.timesteps = sigmas_sub.clone()
            if hasattr(sched, "step_index"):  sched.step_index = 0
            if hasattr(sched, "_step_index"): sched._step_index = 0
        except Exception:
            pass
        return sched, sigmas_sub

    @torch.no_grad()
    def _preview_sample(
        self, batch, num_steps: int, global_step: int = 0,
        Xi_override: torch.Tensor = None,
        noise_override: torch.Tensor = None,
        start_idx_override: Optional[int] = None,
    ):
        H, W = self.cfg.size_h, self.cfg.size_w
        x_in = batch["x_concat_in"].to(self.device, self.dtype)
        m    = batch["m_concat"].to(self.device, self.dtype)

        Xi = (Xi_override.to(self.dtype)
            if Xi_override is not None
            else self._encode_vae_latents(x_in, generator=self._preview_gen, sample=True).to(self.dtype))
        Mi = F.interpolate(m, size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor), mode="nearest").to(self.dtype)
        K  = Mi.float()                  # KEEP
        Hmask = 1.0 - K                  # HOLE
        B = Xi.shape[0]

        sched_full = FM.from_config(self.scheduler.config)
        _, sigmas_full = self._ensure_fm_sigmas(sched_full, max(2, int(num_steps)))
        if sigmas_full.numel() < 2:
            return self._denorm(x_in)

        if start_idx_override is not None:
            start_idx = int(start_idx_override)
        else:
            s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
            N = sigmas_full.numel() - 1
            start_idx = min(max(int((1.0 - s) * N), 0), sigmas_full.numel() - 2)

        sched, sigmas = self._build_preview_scheduler(num_steps, start_idx)
        
        is_strength_max = (float(self.cfg.preview_strength) >= 1.0 - 1e-8)
        
        noise = (noise_override
            if noise_override is not None
            else torch.randn(Xi.shape, dtype=torch.float32, device=Xi.device, generator=self._preview_gen))

        # SD3 inpaint: strength=1.0이면 초기 latents=그 'noise' 자체를 사용
        if is_strength_max:
            z = noise.float()
        else:
            sigma0 = sigmas[0].to(Xi.device, torch.float32)
            z = sched.scale_noise(Xi.float(), sigma0[None], noise).float()

        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer

        was_train = model.training
        model.eval()
        try:
            timesteps = getattr(sched, "sigmas", getattr(sched, "timesteps"))
            n_steps = int(timesteps.numel() - 1)
            for i in range(n_steps):
                sigma_t   = timesteps[i].to(Xi.device, torch.float32)
                sigma_t_b = sigma_t.expand(B)

                # try:
                #     z_in = sched.scale_model_input(z, sigma_t_b)
                # except Exception:
                #     z_in = self._fm_scale(z, sigma_t_b)
                z_in = z

                with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=='cuda')):
                    eps_pred = model(
                        hidden_states=z_in.to(self.dtype),
                        timestep=sigma_t_b,  # FlowMatch uses sigmas vector
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                # step (index-based)
                z = sched.step(eps_pred, sigma_t, z).prev_sample.float()

                # per-step recomposition (공식 inpaint): HOLE만 현재 z 유지, KEEP은 Xi쪽으로 고정
                sigma_next = timesteps[i + 1].to(Xi.device, torch.float32)
                init_latents_proper = sched.scale_noise(Xi.float(), sigma_next[None], noise).float()
                z = K * init_latents_proper + (1.0 - K) * z

            # decode & 최종 픽셀 합성(보기 좋게)
            x_hat = from_latent_sd3(self.vae, z.to(self.dtype))
            K3_keep = m.repeat(1, 3, 1, 1)
            x_final = K3_keep * x_in + (1.0 - K3_keep) * x_hat
            return torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
        finally:
            if was_train:
                model.train()
 
    @torch.no_grad()    
    def one_step_teacher_forcing(
        self, x_in, x_gt, m, sigma0: torch.Tensor, noise: torch.Tensor, recompose=True,
        Xi_override: Optional[torch.Tensor]=None, z0_override: Optional[torch.Tensor]=None
    ):
        """ 학습 분포에서 1-step teacher forcing 확인: ε̂→z0_hat→decode """
        H, W = self.cfg.size_h, self.cfg.size_w
        # SD3와 동일하게 posterior.sample(generator) 사용
        Xi = (Xi_override.float()
            if Xi_override is not None
            else self._encode_vae_latents(x_in, generator=self._preview_gen, sample=True).float())
        z0 = (z0_override.float()
            if z0_override is not None
            else self._encode_vae_latents(x_gt, generator=self._preview_gen, sample=True).float())
        Mi = F.interpolate(m, size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor), mode="nearest").float()
        B = Xi.shape[0]
        sigma_b = sigma0.to(Xi).expand(B)

        Xi_noisy = self.train_scheduler.scale_noise(Xi, sigma_b, noise)
        z0_noisy = self.train_scheduler.scale_noise(z0, sigma_b, noise)
        x_t = Mi * Xi_noisy + (1.0 - Mi) * z0_noisy

        prompt_embeds, pooled = self._encode_prompts(B)
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type=='cuda')):
            eps_pred = (self.transformer.module if isinstance(self.transformer, DDP) else self.transformer)(
                hidden_states=x_t.to(self.dtype), timestep=sigma_b,
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled, return_dict=True
            ).sample.float()

        z0_hat = x_t.float() - sigma_b.view(B,1,1,1) * eps_pred.float()
        x_hat  = from_latent_sd3(self.vae, z0_hat.to(self.dtype))
        if recompose:
            K3 = m.repeat(1, 3, 1, 1)
            x_final = K3 * x_in + (1.0 - K3) * x_hat
            return torch.clamp((x_final + 1.0) * 0.5, 0, 1).cpu()
        else:
            return torch.clamp((x_hat + 1.0) * 0.5, 0, 1).cpu()

    @torch.no_grad()
    def run_infer_once(self, x_in: torch.Tensor, m: torch.Tensor, out_path: str,
                       steps: int = 60, seed: Optional[int] = None) -> torch.Tensor:
        """ 위 프리뷰 루틴과 동일(알파 같은 개념 없음) """
        device = self.device
        H, W = self.cfg.size_h, self.cfg.size_w

        gen = (torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else self._preview_gen)
        Xi = self._encode_vae_latents(x_in.to(self.dtype), generator=gen, sample=True).float()
        Mi = F.interpolate(m, size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor), mode="nearest").float()
        K = Mi
        Hmask = 1.0 - K
        B = Xi.shape[0]

        sched_full = FM.from_config(self.scheduler.config)
        sched, sigmas_full = self._ensure_fm_sigmas(sched_full, max(2, int(steps)))
        s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
        N = sigmas_full.numel() - 1
        start_idx = min(max(int((1.0 - s) * N), 0), sigmas_full.numel() - 2)
        sigmas = sigmas_full[start_idx:].contiguous()
        if sigmas[-1] != 0: sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        try:
            sched.sigmas = sigmas.clone()
            sched.timesteps = sigmas.clone()
            if hasattr(sched, "step_index"):  sched.step_index = 0
            if hasattr(sched, "_step_index"): sched._step_index = 0
        except Exception:
            pass
        
        noise = torch.randn(Xi.shape, dtype=torch.float32, device=Xi.device, generator=gen)
        is_strength_max = (float(self.cfg.preview_strength) >= 1.0 - 1e-8)
        if is_strength_max:
            z = noise.float()   # SD3: strength=1.0이면 초기값=noise
        else:
            s0 = sigmas[0].to(Xi.device, torch.float32)
            z = sched.scale_noise(Xi.float(), s0[None], noise).float()

        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer

        was_train = model.training
        model.eval()
        try:
            timesteps = sched.sigmas
            for i in range(timesteps.numel() - 1):
                sigma_t   = timesteps[i].to(Xi)
                sigma_t_b = sigma_t.expand(B)

                # try:
                #     z_in = sched.scale_model_input(z, sigma_t_b)
                # except Exception:
                #     z_in = self._fm_scale(z, sigma_t_b)
                z_in = z

                with torch.amp.autocast(device_type="cuda", dtype=self.dtype, enabled=(device.type == "cuda")):
                    eps_pred = model(
                        hidden_states=z_in.to(self.dtype),
                        timestep=sigma_t_b,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                z = sched.step(eps_pred, sigma_t, z).prev_sample.float()

                sigma_next = timesteps[i + 1].to(Xi, torch.float32)
                init_latents_proper = sched.scale_noise(Xi.float(), sigma_next[None], noise).float()
                z = K * init_latents_proper + (1.0 - K) * z

            x_hat = from_latent_sd3(self.vae, z.to(self.dtype))
            K3_keep = m.repeat(1, 3, 1, 1)
            x_final = K3_keep * x_in + (1.0 - K3_keep) * x_hat
            save_image(make_grid(x_final.clamp(-1,1).add(1).mul(0.5).cpu(), nrow=1), out_path)
            if hasattr(self, "logger"):
                self.logger.info(f"[infer] saved inference to {out_path}")
            return torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
        finally:
            if was_train:
                model.train()

    @torch.no_grad()
    def _bottom_caption_row(self, names: List[str], widths: List[int],
                            height: int = 28, scale: float = 10.0,
                            bg: float = 1.0, fg: float = 0.0) -> torch.Tensor:
        W_total = int(sum(map(int, widths))); H = int(height * scale)
        bg255 = int(bg * 255); fg255 = int(fg * 255)
        img = Image.new("RGB", (W_total, H), (bg255, bg255, bg255))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(H * 0.7))
        except Exception:
            try: font = ImageFont.load_default()
            except Exception: font = None

        x0 = 0
        for name, w in zip(names, widths):
            w = int(w)
            try:
                bbox = draw.textbbox((0, 0), name, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(name, font=font)
            x = x0 + max(0, (w - tw) // 2)
            y = max(0, (H - th) // 2)
            draw.text((x, y), name, fill=(fg255, fg255, fg255), font=font)
            x0 += w

        t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)

    @torch.no_grad()
    def _save_preview(self, batch: Dict[str, torch.Tensor], global_step: int, max_rows: int = 4):
        if not self.is_main:
            return

        rows = int(self.cfg.get("preview_rows", 1))
        x_in = batch["x_concat_in"][:rows].to(self.device, self.dtype)
        x_gt = batch["x_concat_gt"][:rows].to(self.device, self.dtype)
        m    = batch["m_concat"][:rows].to(self.device, self.dtype)
        if any(t.shape[0] == 0 for t in [x_in, x_gt, m]):
            return

        # 1회 마스크 극성 로그
        if not self._mask_keep_logged:
            keep_ratio = (m[:, :, :, : m.shape[-1] // 2].float().mean().item())
            self.logger.info(f"[debug] left-half KEEP ratio ≈ {keep_ratio:.3f} (Mi=1 KEEP)")
            self._mask_keep_logged = True

        # Xi+noise 미리보기
        num_steps = max(2, int(self.cfg.preview_infer_steps))
        sched_full, sigmas_full = self._set_fm_timesteps(self.scheduler, num_steps)
        s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
        N = sigmas_full.numel() - 1
        start_idx = min(max(int((1.0 - s) * N), 0), sigmas_full.numel() - 2)
        sched_probe, sigmas_probe = self._build_preview_scheduler(num_steps, start_idx)
        # SD3와 동일: posterior.sample + 공유 generator
        Xi_lat = self._encode_vae_latents(x_in, generator=self._preview_gen, sample=True).to(self.dtype)
        z0_lat = self._encode_vae_latents(x_gt, generator=self._preview_gen, sample=True).to(self.dtype)
        sigma0 = sigmas_probe[0].to(Xi_lat.device, torch.float32)
        shared_noise = torch.randn(Xi_lat.shape, dtype=torch.float32, device=Xi_lat.device, generator=self._preview_gen)
        z_init = shared_noise if s >= 1.0 - 1e-8 else sched_probe.scale_noise(Xi_lat.float(), sigma0[None], shared_noise).float()
        xi_noisy_img = from_latent_sd3(self.vae, z_init.to(self.dtype))
        xi_noisy_img = torch.clamp((xi_noisy_img + 1.0) * 0.5, 0.0, 1.0).detach().cpu()

        # inpaint preview        
        pred_img_final = self._preview_sample(
            {"x_concat_in": x_in, "m_concat": m},
            num_steps=num_steps,
            global_step=global_step,
            Xi_override=Xi_lat.float(),
            noise_override=shared_noise,
            start_idx_override=start_idx,
        )

        # one-step TF
        one_step_img = self.one_step_teacher_forcing(
            x_in, x_gt, m,
            sigma0=sigma0, noise=shared_noise, recompose=True,
            Xi_override=Xi_lat.float(), z0_override=z0_lat.float()
        )
        # 패널 빌드
        _, _, Hh, WW = x_gt.shape
        Ww = WW // 2
        person        = x_gt[:, :, :, :Ww]
        garment       = x_gt[:, :, :, Ww:]
        mask_keep     = m[:, :, :, :Ww]
        mask_vis      = mask_keep.repeat(1, 3, 1, 1)
        masked_person = person * mask_keep

        def _cpu32(t): return t.detach().to('cpu', dtype=torch.float32, non_blocking=True).contiguous()

        tiles_and_names = [
            (_cpu32(self._denorm(person))   , "person"),
            (_cpu32(mask_vis)               , "mask_vis"),
            (_cpu32(self._denorm(masked_person)), "masked_person"),
            (_cpu32(self._denorm(garment))  , "garment"),
            (xi_noisy_img                   , "Xi + noise"),
            (_cpu32(self._denorm(x_gt))     , "GT pair"),
            (pred_img_final                 , "preview"),
            (one_step_img                   , "1-step TF"),
        ]

        cols = [t for (t, _) in tiles_and_names]
        panel_bchw = torch.cat(cols, dim=3)
        col_widths = [int(t.shape[-1]) for (t, _) in tiles_and_names]
        col_names  = [name for (_, name) in tiles_and_names]

        grid = make_grid(panel_bchw, nrow=1, padding=0)
        caption_h = int(self.cfg.get("preview_caption_h", 28))
        bottom = self._bottom_caption_row(col_names, col_widths, height=caption_h, scale=2.5, bg=1.0, fg=0.0)[0]
        final_img = torch.cat([grid, bottom], dim=1)

        out_path = os.path.join(self.img_dir, f"step_{global_step:06d}.png")
        save_image(final_img, out_path)
        self.logger.info(f"[img] saved preview at step {global_step}: {out_path}")

    @torch.no_grad()
    def _encode_vae_latents(self, x: torch.Tensor, generator: Optional[torch.Generator] = None, sample: bool = True):
        vdtype = next(self.vae.parameters()).dtype
        posterior = self.vae.encode(x.to(vdtype)).latent_dist
        lat = posterior.sample(generator=generator) if sample else posterior.mean
        sf = self.vae.config.scaling_factor
        sh = getattr(self.vae.config, "shift_factor", 0.0)
        return (lat - sh) * sf

    # ---------------- training core ----------------
    def _sample_sigmas(self, scheduler, B: int) -> torch.Tensor:
        # 스케줄러 내부 timesteps/sigmas까지 실제로 채워 넣음
        _, sigmas = self._ensure_fm_sigmas(scheduler, int(self.cfg.preview_infer_steps))
        idx = torch.randint(0, sigmas.numel() - 1, (B,), device=self.device)
        return sigmas.index_select(0, idx).to(self.device)

    def step(self, batch, global_step: int):
        H, W = self.cfg.size_h, self.cfg.size_w
        x_concat_in = batch["x_concat_in"].to(self.device, self.dtype)
        x_concat_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m_concat = batch["m_concat"].to(self.device, self.dtype)

        with torch.no_grad():
            Xi = self._encode_vae_latents(x_concat_in, generator=None, sample=True).to(self.dtype)
            z0 = self._encode_vae_latents(x_concat_gt, generator=None, sample=True).to(self.dtype)
            Mi = F.interpolate(m_concat, size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor), mode="nearest").to(self.dtype)  # [B,1,h,w]

        B = Xi.shape[0]
        K = Mi.float()        # KEEP
        Hmask = 1.0 - K       # HOLE
        
        if (self.is_main and (not self._dumped_mask_once) and global_step == 0):
            # latent 해상도 마스크를 픽셀 해상도로 업샘플해서 보이기 쉽게 저장
            Mi_pix    = F.interpolate(Mi,    size=(H, 2 * W), mode="nearest").clamp(0, 1)
            Hmask_pix = (1.0 - Mi).clamp(0, 1)
            Hmask_pix = F.interpolate(Hmask_pix, size=(H, 2 * W), mode="nearest").clamp(0, 1)

            vis_keep = Mi_pix[:1].repeat(1, 3, 1, 1)     # 흰색=KEEP
            vis_hole = Hmask_pix[:1].repeat(1, 3, 1, 1)  # 흰색=HOLE

            save_image(vis_keep, os.path.join(self.img_dir, "debug_mask_keep_pixel.png"))
            save_image(vis_hole, os.path.join(self.img_dir, "debug_mask_hole_pixel.png"))

            # 원본 입력과도 나란히 보고 싶으면(선택):
            # save_image(self._denorm(x_concat_gt[:1]), os.path.join(self.img_dir, "debug_gt_pair.png"))

            self.logger.info("[debug] dumped debug_mask_keep_pixel.png / debug_mask_hole_pixel.png")
            self._dumped_mask_once = True

        # sigma sample
        sigmas = self._sample_sigmas(self.train_scheduler, B)      # [B] (mu 반영된 스케줄에서 샘플)
        noise  = torch.randn_like(z0, dtype=torch.float32)

        # 동일 난수를 Xi/z0에 적용하여 둘 다 FM 방식으로 노이징
        Xi_noisy = self.train_scheduler.scale_noise(Xi.float(), sigmas, noise)
        z0_noisy = self.train_scheduler.scale_noise(z0.float(), sigmas, noise)
        x_t_mixed = K * Xi_noisy + Hmask * z0_noisy

        target = noise  # ε-target

        # try:
        #     z_in = self.train_scheduler.scale_model_input(x_t_mixed, sigmas)
        # except Exception:
        #     z_in = self._fm_scale(x_t_mixed, sigmas)
        z_in = x_t_mixed

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(B)

        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
            model = self.transformer
            eps_pred = model(
                hidden_states=z_in.to(self.dtype),
                timestep=sigmas,    # FlowMatch uses sigmas as "t"
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=True,
            ).sample

        eps_pred_f32 = eps_pred.float()

        # --------- Losses ---------
        # ε-MSE: HOLE만 (면적 정규화)
        C = z0.shape[1]
        denom = (Hmask.sum() * C).clamp_min(1.0)
        if bool(self.cfg.get("loss_sigma_weight", False)):
            sigma_b11 = sigmas.view(B, 1, 1, 1).float()
            w = 1.0 / (sigma_b11 ** 2 + 1.0)
            loss_eps = ((((eps_pred_f32 - target) ** 2) * Hmask) * w).sum() / denom
        else:
            loss_eps = (((eps_pred_f32 - target) ** 2) * Hmask).sum() / denom

        # latent x0 재구성: HOLE만 (선택)
        z0_hat = x_t_mixed.float() - sigmas.view(B,1,1,1).float() * eps_pred_f32
        rec_lambda = float(self.cfg.latent_rec_lambda)
        warmup_steps = int(getattr(self.cfg, "latent_rec_lambda_warmup_steps", 0))
        if warmup_steps > 0 and global_step < warmup_steps:
            rec_lambda_eff = 0.0
        else:
            rec_lambda_eff = rec_lambda
        loss_lat = ((z0_hat - z0.float()) ** 2 * Hmask).sum() / Hmask.sum().clamp_min(1.0)

        # KEEP consistency (옵션, 기본 0)
        keep_lambda = float(self.cfg.get("keep_consistency_lambda", 0.0))
        loss_keep = torch.tensor(0.0, device=self.device)
        if keep_lambda > 0.0:
            loss_keep = (((z0_hat - z0.float()) ** 2) * K).sum() / K.sum().clamp_min(1.0)

        # Garment recon (옵션, 기본 0)
        gar_lambda = float(self.cfg.get("garment_rec_lambda", 0.0))
        loss_gar = torch.tensor(0.0, device=self.device)
        if gar_lambda > 0.0:
            Wlat = Mi.shape[-1]
            Rmask = torch.zeros_like(Mi); Rmask[..., :, :, Wlat//2:] = 1.0
            loss_gar = (((z0_hat - z0.float()) ** 2) * Rmask).sum() / Rmask.sum().clamp_min(1.0)

        loss_total = loss_eps + rec_lambda_eff * loss_lat + keep_lambda * loss_keep + gar_lambda * loss_gar
        if not torch.isfinite(loss_total) or not loss_total.requires_grad:
            return None

        return (
            loss_total,
            float(loss_eps.detach().cpu()),
            float(loss_lat.detach().cpu()),
            float(loss_keep.detach().cpu()),
            float(loss_gar.detach().cpu()),
        )

    def _save_ckpt(self, epoch: int, train_loss_epoch: float, global_step: int) -> str:
        ckpt_path = os.path.join(self.model_dir, f"epoch_{epoch}_loss_{train_loss_epoch:.04f}.ckpt")
        payload = {
            "transformer": ddp_state_dict(self.transformer),
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

    def _resume_from_ckpt_if_needed(self, resume_path: Optional[str]):
        if not resume_path or not os.path.isfile(resume_path):
            return
        ckpt = torch.load(resume_path, map_location="cpu")
        try:
            self.transformer.load_state_dict(ckpt["transformer"], strict=False)
        except Exception:
            self.transformer.load_state_dict(ckpt["transformer"], strict=False)

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.start_step  = int(ckpt.get("global_step", 0))
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            dev = next(self.transformer.parameters()).device
            for s in self.optimizer.state.values():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        s[k] = v.to(dev)
        if self.is_main:
            self.logger.info(f"[resume] loaded {resume_path} | start_epoch={self.start_epoch}, start_step={self.start_step}")
            print(f"[resume] loaded {resume_path} | start_epoch={self.start_epoch}, start_step={self.start_step}")

    def _grad_norm_sum(self, module: nn.Module) -> float:
        s = 0.0
        for p in module.parameters():
            if p.requires_grad and (p.grad is not None):
                try: s += float(p.grad.detach().float().norm().cpu())
                except Exception: pass
        return s

    def train(self):
        global_step = getattr(self, "start_step", 0)
        epoch = getattr(self, "start_epoch", 0)
        self.transformer.train()

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

        # meters
        comp_eps_sum = comp_lat_sum = comp_keep_sum = comp_gar_sum = 0.0

        while global_step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            comp_eps_accum = comp_lat_accum = comp_keep_accum = comp_gar_accum = 0.0
            nonfinite = False
            reason = ""

            for _ in range(self.cfg.grad_accum):
                batch = next(data_iter)
                out = self.step(batch, global_step)
                if (out is None):
                    nonfinite = True; reason = "None/NaN in forward"; break
                (loss, loss_eps_val, loss_lat_val, loss_keep_val, loss_gar_val) = out

                if (not torch.isfinite(loss)) or (not loss.requires_grad):
                    nonfinite = True; reason = "non-finite/detached loss"; break

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += float(loss.detach().cpu())
                comp_eps_accum  += float(loss_eps_val)
                comp_lat_accum  += float(loss_lat_val)
                comp_keep_accum += float(loss_keep_val)
                comp_gar_accum  += float(loss_gar_val)

            if nonfinite:
                msg = f"[warn] skipping step {global_step}: {reason}."
                if self.is_main: pbar.write(msg)
                self.logger.info(msg)
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_scaler: self.scaler.update()
                global_step += 1
                if self.is_main: pbar.update(1)
                continue

            # grad norms
            tf_grad_norm = self._grad_norm_sum(self.transformer)

            # clip & step
            if self.use_scaler:
                try:
                    self.scaler.unscale_(self.optimizer)
                except ValueError:
                    pass
            torch.nn.utils.clip_grad_norm_((p for p in self.transformer.parameters() if p.requires_grad), 0.5)
            if self.use_scaler:
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                self.optimizer.step()

            global_step += 1
            if self.is_main: pbar.update(1)

            # meters update
            self._epoch_loss_sum += loss_accum / max(1, self.cfg.grad_accum)
            self._epoch_loss_count += 1
            comp_eps_sum  += comp_eps_accum  / max(1, self.cfg.grad_accum)
            comp_lat_sum  += comp_lat_accum  / max(1, self.cfg.grad_accum)
            comp_keep_sum += comp_keep_accum / max(1, self.cfg.grad_accum)
            comp_gar_sum  += comp_gar_accum  / max(1, self.cfg.grad_accum)

            train_loss_avg = self._epoch_loss_sum / max(1, self._epoch_loss_count)
            loss_eps_avg   = comp_eps_sum / max(1, self._epoch_loss_count)
            loss_lat_avg   = comp_lat_sum / max(1, self._epoch_loss_count)
            loss_keep_avg  = comp_keep_sum / max(1, self._epoch_loss_count)
            loss_gar_avg   = comp_gar_sum / max(1, self._epoch_loss_count)

            if self.is_main and ((global_step % self.cfg.log_every) == 0 or global_step == 1):
                self.tb.add_scalar("train/loss_total", train_loss_avg, global_step)
                self.tb.add_scalar("train/loss_eps",   loss_eps_avg,   global_step)
                self.tb.add_scalar("train/loss_lat",   loss_lat_avg,   global_step)
                self.tb.add_scalar("train/loss_keep",  loss_keep_avg,  global_step)
                self.tb.add_scalar("train/loss_gar",   loss_gar_avg,   global_step)
                self.tb.add_scalar("train/grad_norm_transformer", tf_grad_norm, global_step)

                prog = (global_step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0.0
                pct = int(prog * 100)
                line = (f"Epoch {epoch}: {pct:3d}% | step {global_step}/{self.cfg.max_steps} "
                        f"| total={train_loss_avg:.4f} | eps={loss_eps_avg:.4f} | lat={loss_lat_avg:.4f} | keep={loss_keep_avg:.4f} | gar={loss_gar_avg:.4f}")
                pbar.set_postfix_str(f"tot={train_loss_avg:.4f}")
                pbar.write(line); self.logger.info(line)

            if self.is_dist:
                dist.barrier()

            if self.is_main and ((global_step % self.cfg.image_every) == 0 or global_step == 1):
                try:
                    batch_vis = next(data_iter)
                    self._save_preview(batch_vis, global_step, max_rows=min(4, self.cfg.batch_size))
                    pbar.write(f"[img] saved preview at step {global_step}")
                except Exception as e:
                    msg = f"[warn] preview save failed: {e}"
                    pbar.write(msg); self.logger.info(msg)

            if self.is_dist:
                dist.barrier()

            if (global_step % self.steps_per_epoch) == 0:
                epoch += 1
                if self.is_main and (epoch % self.cfg.save_epoch_ckpt) == 0:
                    path = self._save_ckpt(epoch, train_loss_avg, global_step)
                    pbar.write(f"[save-epoch] {path}")
                self._epoch_loss_sum = 0.0
                self._epoch_loss_count = 0
                comp_eps_sum = comp_lat_sum = comp_keep_sum = comp_gar_sum = 0.0
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
    "mask_based": True, "invert_mask": True,

    "lr": 5e-5,
    "batch_size": 8, "grad_accum": 1, "max_steps": 128000,
    "seed": 1337, "num_workers": 4,
    "mixed_precision": "fp16",
    "use_scaler": True,

    "prefer_xformers": True,          # (미사용 플래그; 필요 시 확장)
    "disable_text": True,             # 무의미 프롬프트(uncond) 사용
    "save_root_dir": "logs", "save_name": "catvton_sd35_inpaint",
    "log_every": 50, "image_every": 500,
    "save_epoch_ckpt": 15,
    "default_resume_ckpt": None,
    "resume_ckpt": None,

    # inpainting objective
    "latent_rec_lambda": 0.15,
    "latent_rec_lambda_warmup_steps": 4000,
    "loss_sigma_weight": True,

    # (선택) 부가 항목 — 기본 0
    "keep_consistency_lambda": 0.0,
    "garment_rec_lambda": 0.0,

    # preview
    "preview_infer_steps": 60,
    "preview_seed": 1234,
    "preview_strength": 0.3,
    "preview_rows" : 4,
    "preview_caption_h": 28,

    # runtime
    "hf_token": None,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/stage1.yaml", help="YAML config path. CLI overrides YAML.")
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
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--image_every", type=int, default=None)
    p.add_argument("--mixed_precision", type=str, default=None, choices=["fp16", "fp32", "bf16"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--hf_token", type=str, default=None)

    p.add_argument("--disable_text", action="store_true")
    p.add_argument("--enable_text", action="store_true")

    p.add_argument("--save_root_dir", type=str, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--preview_infer_steps", type=int, default=None)
    p.add_argument("--preview_strength", type=float, default=None)
    p.add_argument("--preview_seed", type=int, default=None)
    p.add_argument("--resume_ckpt", type=str, default=None)

    p.add_argument("--latent_rec_lambda", type=float, default=None)
    p.add_argument("--latent_rec_lambda_warmup_steps", type=int, default=None)
    p.add_argument("--loss_sigma_weight", action="store_true")
    p.add_argument("--no_loss_sigma_weight", action="store_true")

    p.add_argument("--keep_consistency_lambda", type=float, default=None)
    p.add_argument("--garment_rec_lambda", type=float, default=None)

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

    if getattr(args, "enable_text", False):
        cfg["disable_text"] = False
    elif getattr(args, "disable_text", False):
        cfg["disable_text"] = True

    if getattr(args, "no_loss_sigma_weight", False):
        cfg["loss_sigma_weight"] = False
    elif getattr(args, "loss_sigma_weight", False):
        cfg["loss_sigma_weight"] = True

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
