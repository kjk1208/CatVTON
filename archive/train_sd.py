# train_sd35.py — SD3.5 CatVTON (DDP-ready, true inpainting; NO projector)
# - Train: FlowMatch (official SD3-style sampling: indices→timesteps→sigmas, noisy_model_input) + official SD3 loss
# - Infer/Preview: FlowMatch Euler + per-step recomposition (official inpaint style)
# - Text: unconditional embeds only (no CLIP/T5 text usage by user; we still pull uncond from pipeline)
# - Dataset/preview/IO format: preserved from user's original script
# - Freeze: self-attn Q/K/V/Out only (CatVTON-style)
#
# Notes:
#  • Per your request, we mirror the official train_controlnet_sd3.py variable usage:
#      model_input, noise, bsz, indices, timesteps, sigmas, noisy_model_input
#    and the official SD3 loss (weighting + optional preconditioning).
#  • We still form an inpainting input by mixing {Xi_noisy, z0_noisy} with the KEEP mask,
#    but "noisy_model_input" is used *exactly* as the model input variable (same name & role as official).
#  • Mixed precision handling mirrors the official approach (fp16/bf16/fp32). If you don't specify,
#    fp16 is a good default on most NVIDIA GPUs; bf16 is recommended on Ampere+ when stable.

import os
import re
import json
import random
import argparse
import copy
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
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as FM
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
except Exception as e:
    raise ImportError(
        "This training script requires diffusers with SD3 support. "
        "Install/update: pip install -U 'diffusers>=0.36.0' 'transformers>=4.43.0' 'safetensors>=0.4.3'"
    ) from e

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
def freeze_all_but_self_attn_qkv(transformer, open_to_add_out: bool=False, open_io: bool=False, open_ffn_blocks: list[int]=None, open_block_norms: bool=False):
    # 0) freeze all
    for p in transformer.parameters():
        p.requires_grad = False

    # 1) self-attn Q/K/V/Out trainable
    for name, m in transformer.named_modules():
        if name.endswith(".attn") or hasattr(m, "to_q"):
            for attr in ["to_q","to_k","to_v","to_out"]:
                sub = getattr(m, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad = True

    # 2) cross path: keep frozen; optionally open to_add_out
    for name, m in transformer.named_modules():
        if name.endswith(".attn"):
            for attr in ["add_k_proj","add_v_proj","add_q_proj"]:
                sub = getattr(m, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad = False
            sub = getattr(m, "to_add_out", None)
            if sub is not None:
                for p in sub.parameters():
                    p.requires_grad = bool(open_to_add_out)

    # 3) I/O adapters
    if open_io:
        if hasattr(transformer, "pos_embed") and hasattr(transformer.pos_embed, "proj"):
            for p in transformer.pos_embed.proj.parameters():
                p.requires_grad = True
        for p in transformer.norm_out.parameters():
            p.requires_grad = True
        for p in transformer.proj_out.parameters():
            p.requires_grad = True

     # 4) FFN 일부 블록 개방 (+선택적으로 블록 내 norm)
    if open_ffn_blocks is not None:
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is not None:
            for idx in open_ffn_blocks:
                if idx < 0 or idx >= len(blocks): continue
                blk = blocks[idx]
                for attr in ["ff", "ff_context"]:
                    sub = getattr(blk, attr, None)
                    if sub is not None:
                        for p in sub.parameters(): p.requires_grad = True
                if open_block_norms:
                    for attr in ["norm2", "norm2_context"]:
                        sub = getattr(blk, attr, None)
                        if sub is not None:
                            for p in sub.parameters(): p.requires_grad = True

    # safety
    for n, p in transformer.named_parameters():
        if (".attn.add_" in n) and (".attn.to_add_out" not in n) and p.requires_grad:
            raise RuntimeError(f"Cross add path unexpectedly trainable: {n}")

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    kept = [n for n, p in transformer.named_parameters() if p.requires_grad][:16]
    if trainable == 0:
        raise RuntimeError("No params unfrozen; check patterns.")
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

def _try_copy_running_script(dst_dir: str):
    """
    현재 실행 중인 트레이닝 스크립트를 dst_dir로 복사합니다.
    torchrun 다중 프로세스 환경에서는 rank==0 에서만 호출하세요.
    """
    import shutil
    try:
        # sys.argv[0]이 가장 안전하게 "실행한 파일"을 가리킴
        script_path = os.path.abspath(sys.argv[0])
        if os.path.isfile(script_path):
            base = os.path.basename(script_path)
            dst_path = os.path.join(dst_dir, base)
            shutil.copy2(script_path, dst_path)
    except Exception as e:
        print(f"[warn] failed to copy running script: {e}")



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
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # ---- Schedulers ----
        self.train_scheduler = FM.from_config(pipe.scheduler.config)  # used to get base config
        self.noise_scheduler_copy = copy.deepcopy(self.train_scheduler)  # for training-time get_sigmas()
        self.scheduler = FM.from_config(pipe.scheduler.config)  # inference/preview
        pred_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        self.train_scheduler.config.prediction_type = pred_type
        self.scheduler.config.prediction_type = pred_type
    

        # >>> ensure FM sigmas available
        train_steps = int(getattr(self.cfg, "train_sched_steps", self.noise_scheduler_copy.config.num_train_timesteps))
        self._ensure_fm_sigmas(self.noise_scheduler_copy, train_steps)
        self._ensure_fm_sigmas(self.train_scheduler,      train_steps)
        
        self.noise_scheduler_copy.config.num_train_timesteps = train_steps
        self.train_scheduler.config.num_train_timesteps       = train_steps
        
        if self.is_main:
            self.logger.info(f"[sched] training/inference = FlowMatchEuler (pred={pred_type})")
            self.logger.info(f"Loaded SD3 model: {cfg.sd3_model}")

        # memory knobs
        try:
            self.transformer.enable_gradient_checkpointing()
            if self.is_main: print("[mem] gradient checkpointing ON")
        except Exception as e:
            if self.is_main: print(f"[mem] gradient checkpointing not available: {e}")

        # only self attention 20250916
        # Freeze except self-attn Q/K/V/Out
        tf = self.transformer  # DDP 래핑 전
        N  = len(tf.transformer_blocks)
        mid_ffn_blocks = list(range(N//3, min(N, 2*N//3 + 2)))

        trainable_tf, keep_names_sample = freeze_all_but_self_attn_qkv(transformer=self.transformer, open_io=True, open_ffn_blocks=mid_ffn_blocks, open_block_norms=False)
        if self.is_main:
            sanity_check_self_vs_cross(self.transformer, tag="init_before_ddp")
        if self.dtype == torch.float16:
            # keep trainables in fp32 for stability when using fp16 AMP
            for _, p in self.transformer.named_parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)

        if self.is_main:
            print(f"Trainable params (transformer only)={trainable_tf/1e6:.2f}M")
            _debug_print_trainables(self.transformer, "after_freeze_before_ddp")
        # only self attention 20250916 


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
            print(f"[amp] dtype={self.dtype}, use_scaler={self.use_scaler} (bf16 recommended on Ampere+/Hopper)")

        # DDP wrap
        #self attention only 20250916
        self.transformer = DDP(
            self.transformer, device_ids=[self.local_rank], output_device=self.local_rank,
            broadcast_buffers=False, find_unused_parameters=False
        )
        if self.is_main:
            sanity_check_self_vs_cross(self.transformer.module, tag="init_after_ddp")
            _debug_print_trainables(self.transformer.module, "after_ddp")
        #self attention only 20250916        
        
        # self attention only 20250916
        # optimizer: param groups

        qkv_params, add_out_params, io_params, ffn_params = [], [], [], []
        for n, p in self.transformer.named_parameters():
            if not p.requires_grad:
                continue
            if (".attn.to_q" in n) or (".attn.to_k" in n) or (".attn.to_v" in n) or (".attn.to_out" in n):
                qkv_params.append(p)
                if self.is_main:
                    print(f"QKV param: {n}")
                    self.logger.info(f"QKV param: {n}")                
            elif ".attn.to_add_out" in n:
                add_out_params.append(p)
                if self.is_main:
                    print(f"Add out param: {n}")
                    self.logger.info(f"Add out param: {n}")                
            elif ("pos_embed.proj" in n) or ("norm_out" in n) or ("proj_out" in n):
                io_params.append(p)
                if self.is_main:
                    print(f"IO param: {n}")
                    self.logger.info(f"IO param: {n}")                
            elif (".ff." in n) or (".ff_context." in n):  # ★ FFN
                ffn_params.append(p)
                if self.is_main:
                    print(f"FFN param: {n}")
                    self.logger.info(f"FFN param: {n}")
                

        #assert len(io_params) == 0, "IO adapters should be frozen (pos_embed.proj/norm_out/proj_out)"
        assert len(add_out_params) == 0, "to_add_out/add_* should be frozen"

        param_groups = []
        if qkv_params:
            param_groups.append({"params": qkv_params, "lr": cfg.lr})
        if io_params:
            param_groups.append({"params": io_params, "lr": cfg.lr * 2.0})
        if add_out_params:
            param_groups.append({"params": add_out_params, "lr": cfg.lr * 0.5})
        if ffn_params:
            param_groups.append({"params": ffn_params, "lr": cfg.lr * 0.5})
        if not param_groups:
            tf_params = [p for p in self.transformer.parameters() if p.requires_grad]
            param_groups = [{"params": tf_params, "lr": cfg.lr}]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

        # self attention only 20250916

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
        

    def _encode_prompts(self, bsz: int):
        """
        고정 프롬프트를 반드시 str로 만들어 파이프라인에 전달.
        encode_prompt는 str 또는 list[str]를 받는데, 여기서는 str을 사용.
        """
        # 1) 완전 비활성화: 파이프라인에서 '빈 프롬프트'로 얻은 uncond 임베딩만 사용
        if getattr(self.cfg, "disable_text", False):
            if not hasattr(self, "_uncond_pe"):
                pe, _, ppe, _ = self.encode_prompt(
                    prompt="", prompt_2="", prompt_3="",
                    device=self.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                self._uncond_pe  = pe.detach().to(self.dtype)
                self._uncond_ppe = ppe.detach().to(self.dtype)

            return (self._uncond_pe.expand(bsz, -1, -1).contiguous(),
                    self._uncond_ppe.expand(bsz, -1).contiguous())

        fixed_txt = getattr(
            self.cfg,
            "fixed_prompt",
            "high-resolution, high-quality, highly detailed image; natural lighting; sharp focus"
        )
        # 어떤 입력이 와도 문자열이 되게 강제
        if isinstance(fixed_txt, (list, tuple)):
            fixed_txt = " ".join(map(str, fixed_txt))
        elif not isinstance(fixed_txt, str):
            fixed_txt = str(fixed_txt)

        if not hasattr(self, "_fixed_pe"):
            pe, _, ppe, _ = self.encode_prompt(
                prompt=fixed_txt,        # ← 리스트 대신 문자열
                prompt_2=fixed_txt,
                prompt_3=fixed_txt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            self._fixed_pe  = pe.detach().to(self.dtype)
            self._fixed_ppe = ppe.detach().to(self.dtype)

        return (self._fixed_pe.expand(bsz, -1, -1).contiguous(),
                self._fixed_ppe.expand(bsz, -1).contiguous())



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
        tf = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
        patch = getattr(tf, "config").patch_size
        vae_sf = self.vae_scale_factor

        use_ds = (not getattr(self.cfg, "disable_dynamic_shifting", False)) and \
                bool(getattr(sched.config, "use_dynamic_shifting", False))
        if use_ds:
            image_seq_len = (H // vae_sf // patch) * ((2 * W) // vae_sf // patch)  # width doubled (concat)
            base_len  = getattr(sched.config, "base_image_seq_len", 256)
            max_len   = getattr(sched.config, "max_image_seq_len", 4096)
            base_s    = getattr(sched.config, "base_shift", 0.5)
            max_s     = getattr(sched.config, "max_shift", 1.16)
            mu = base_s + (max_s - base_s) * (image_seq_len - base_len) / max(1, (max_len - base_len))
            kwargs["mu"] = float(mu)

        sched.set_timesteps(num_steps, device=self.device, **kwargs)
        sigmas = sched.timesteps
        if sigmas[-1] != 0:
            sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
            sched.timesteps = sigmas
        sched.sigmas = sigmas
        return sched, sigmas

    def _ensure_fm_sigmas(self, scheduler, num_steps: int):
        sched, sigmas = self._set_fm_timesteps(scheduler, num_steps)
        
        # 시그마가 내림차순인지 확인
        if sigmas[0] < sigmas[-1]:  # 만약 오름차순이면
            print("sigmas가 오름차순!!!!")
        try:
            scheduler.sigmas    = sigmas.clone().to(torch.float32)
            scheduler.timesteps = sigmas.clone().to(torch.float32)
        except Exception:
            pass
        return sched, sigmas

    # OFFICIAL-style get_sigmas(t)
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---------- NEW: unified conversion helper ----------
    @staticmethod
    def _raw_to_eps_x0(raw: torch.Tensor, z_t: torch.Tensor, sigma4: torch.Tensor, pred_type: str):
        """
        Convert raw model output to (epsilon, x0) consistently for FM schedulers.
        raw      : [B,C,H,W] model output
        z_t      : [B,C,H,W] current noisy latents x_t
        sigma4   : [B,1,1,1] sigma broadcast tensor
        pred_type: 'epsilon' or 'x0'/'sample'. 'v_prediction' is intentionally not wired.
        """
        pt = str(pred_type).lower()
        if pt == "epsilon":
            eps = raw.float()
            x0  = z_t.float() - sigma4.float() * eps
            return eps, x0
        if pt in ("x0", "sample"):
            x0  = raw.float()
            eps = (z_t.float() - x0) / sigma4.clamp_min(1e-8).float()
            return eps, x0
        if pt in ("v_prediction", "v-pred", "v"):
            # Safer to fail than silently be wrong; SD3.5 FM usually does not use v-pred.
            raise NotImplementedError("v_prediction mapping is model/scheduler-specific. Use 'epsilon' or 'sample(x0)'.")
        # default fallback: treat as epsilon
        eps = raw.float()
        x0  = z_t.float() - sigma4.float() * eps
        return eps, x0

    # ---------------- preview / inference ----------------
    @torch.no_grad()
    def _build_preview_scheduler(self, num_steps: int, start_idx: int):
        steps = max(2, int(num_steps))
        sched_full, sigmas_full = self._set_fm_timesteps(self.scheduler, steps)

        start_idx = min(max(0, int(start_idx)), int(sigmas_full.numel()) - 2)
        sigmas_sub = sigmas_full[start_idx:].contiguous()
        if sigmas_sub[-1] != 0:
            sigmas_sub = torch.cat([sigmas_sub, sigmas_sub.new_zeros(1)])

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
        self,
        num_steps: int,
        start_latents: torch.Tensor,         # ← 시작 latent를 명시: Xi_lat 또는 z0_lat
        pixels_for_keep: torch.Tensor,       # ← 합성에 쓸 픽셀: x_in 또는 x_gt
        m: torch.Tensor,                     # [B,1,H,2W]
        shared_noise: Optional[torch.Tensor] = None,
        start_idx_override: Optional[int] = None,
    ):
        H, W = self.cfg.size_h, self.cfg.size_w
        device = self.device
        dtype  = self.dtype

        # 마스크를 latent 해상도로
        Mi = F.interpolate(
            m.to(device, dtype),
            size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor),
            mode="nearest"
        ).float()
        K = Mi
        B = start_latents.shape[0]

        # 스케줄러/시그마 설정
        sched_full = FM.from_config(self.scheduler.config)
        _, sigmas_full = self._ensure_fm_sigmas(sched_full, max(2, int(num_steps)))
        if sigmas_full.numel() < 2:
            return torch.clamp((pixels_for_keep.to(device, dtype) + 1.0) * 0.5, 0.0, 1.0)

        if start_idx_override is None:
            s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
            N = sigmas_full.numel() - 1
            start_idx = min(max(int((1.0 - s) * N), 0), sigmas_full.numel() - 2)
        else:
            start_idx = int(start_idx_override)

        sched, timesteps = self._build_preview_scheduler(num_steps, start_idx)

        # 공유 노이즈
        if shared_noise is None:
            shared_noise = torch.randn_like(start_latents, dtype=torch.float32, device=device, generator=self._preview_gen)

        # 시작 latent는 호출자가 넘긴 값(start_latents)을 사용
        sigma0 = timesteps[0].to(device, torch.float32)
        start_noisy = sched.scale_noise(start_latents.float(), sigma0[None], shared_noise).float()
        z = K * start_noisy + (1.0 - K) * (sigma0 * shared_noise).float()

        # 프롬프트 임베딩
        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer

        was_train = model.training
        model.eval()
        try:
            pred_type = str(getattr(self.scheduler.config, "prediction_type", "epsilon")).lower()
            for i in range(int(timesteps.numel() - 1)):
                sigma_t   = timesteps[i].to(device, torch.float32)
                sigma_t_b = sigma_t.repeat(B)
                sigma4    = sigma_t.view(1,1,1,1).expand(B,1,1,1).float()

                z_in = self._fm_scale(z, sigma_t_b)
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device.type=='cuda')):
                    raw = model(
                        hidden_states=z_in.to(dtype),
                        timestep=sigma_t_b,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                # raw를 ε/x0로 동시 변환하고, 스케줄러가 기대하는 쪽을 넘긴다.
                eps_pred, x0_pred = self._raw_to_eps_x0(raw, z, sigma4, pred_type)
                model_output_for_step = eps_pred if pred_type == "epsilon" else x0_pred

                z = sched.step(model_output_for_step, sigma_t, z).prev_sample.float()

                # inpaint recomposition (동일)
                sigma_next = timesteps[i + 1].to(device, torch.float32)
                init_latents_proper = sched.scale_noise(start_latents.float(), sigma_next[None], shared_noise).float()
                z = K * init_latents_proper + (1.0 - K) * z

            # 디코드 + 픽셀 합성
            x_hat = from_latent_sd3(self.vae, z.to(dtype))
            K3_keep = m.to(device, dtype).repeat(1, 3, 1, 1)
            x_final = K3_keep * pixels_for_keep.to(device, dtype) + (1.0 - K3_keep) * x_hat
            x_final = torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
            return x_final
        finally:
            if was_train:
                model.train()

    @torch.no_grad()
    def run_infer_once(self, x_in: torch.Tensor, m: torch.Tensor, out_path: str,
                       steps: int = 60, seed: Optional[int] = None) -> torch.Tensor:
        device = self.device
        H, W = self.cfg.size_h, self.cfg.size_w

        gen = (torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else self._preview_gen)
        Xi = self._encode_vae_latents(x_in.to(self.dtype), generator=gen, sample=True).float()
        Mi = F.interpolate(m, size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor), mode="nearest").float()
        K = Mi
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
        s0 = sigmas[0].to(Xi.device, torch.float32)
        Xi_noisy0 = sched.scale_noise(Xi.float(), s0[None], noise).float()
        z = K * Xi_noisy0 + (1.0 - K) * (s0 * noise).float()

        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer

        was_train = model.training
        model.eval()
        try:
            timesteps = sched.sigmas
            pred_type = str(getattr(self.scheduler.config, "prediction_type", "epsilon")).lower()
            for i in range(timesteps.numel() - 1):
                sigma_t   = timesteps[i].to(Xi)
                sigma_t_b = sigma_t.repeat(B)
                sigma4    = sigma_t.view(1,1,1,1).expand(B,1,1,1).float()

                z_in = self._fm_scale(z, sigma_t_b)
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype, enabled=(device.type == "cuda")):
                    raw = model(
                        hidden_states=z_in.to(self.dtype),
                        timestep=sigma_t_b,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                eps_pred, x0_pred = self._raw_to_eps_x0(raw, z, sigma4, pred_type)
                model_output_for_step = eps_pred if pred_type == "epsilon" else x0_pred

                z = sched.step(model_output_for_step, sigma_t, z).prev_sample.float()

                # inpaint recomposition (동일)
                sigma_next = timesteps[i + 1].to(Xi, torch.float32)
                init_latents_proper = sched.scale_noise(Xi.float(), sigma_next[None], noise).float()
                z = K * init_latents_proper + (1.0 - K) * z

            x_hat = from_latent_sd3(self.vae, z.to(self.dtype))
            K3_keep = m.repeat(1, 3, 1, 1)
            x_final = K3_keep * x_in + (1.0 - K3_keep) * x_hat
            x_final = torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
            save_image(make_grid(x_final.cpu(), nrow=1), out_path)
            if hasattr(self, "logger"):
                self.logger.info(f"[infer] saved inference to {out_path}")
            return x_final
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

        if not self._mask_keep_logged:
            keep_ratio = (m[:, :, :, : m.shape[-1] // 2].float().mean().item())
            self.logger.info(f"[debug] left-half KEEP ratio ≈ {keep_ratio:.3f} (Mi=1 KEEP)")
            self._mask_keep_logged = True

        num_steps = max(2, int(self.cfg.preview_infer_steps))
        sched_full, sigmas_full = self._set_fm_timesteps(self.scheduler, num_steps)
        s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
        N = sigmas_full.numel() - 1
        start_idx = min(max(int((1.0 - s) * N), 0), sigmas_full.numel() - 2)
        sched_probe, sigmas_probe = self._build_preview_scheduler(num_steps, start_idx)

        Xi_lat = self._encode_vae_latents(x_in, generator=self._preview_gen, sample=True).to(self.dtype)
        z0_lat = self._encode_vae_latents(x_gt, generator=self._preview_gen, sample=True).to(self.dtype)
        sigma0 = sigmas_probe[0].to(Xi_lat.device, torch.float32)
        shared_noise = torch.randn(Xi_lat.shape, dtype=torch.float32, device=Xi_lat.device, generator=self._preview_gen)

        Mi = F.interpolate(
            m, size=(self.cfg.size_h // self.vae_scale_factor, (2 * self.cfg.size_w) // self.vae_scale_factor),
            mode="nearest"
        ).float()
        Xi_noisy0 = sched_probe.scale_noise(Xi_lat.float(), sigma0[None], shared_noise).float()
        
        z_init_vis = Mi * Xi_noisy0 + (1.0 - Mi) * (sigma0 * shared_noise)
        xi_noisy_img = from_latent_sd3(self.vae, z_init_vis.to(self.dtype))
        xi_noisy_img = torch.clamp((xi_noisy_img + 1.0) * 0.5, 0.0, 1.0).detach().cpu()

        # 두 가지 프리뷰(입력 합성, GT 합성) 받기
        preview_from_input = self._preview_sample(
            num_steps=num_steps,
            start_latents=Xi_lat.float(),
            pixels_for_keep=x_in,
            m=m,
            shared_noise=shared_noise,
            start_idx_override=start_idx,
        )

        # (B) GT(latents=z0)로부터 시작, GT 픽셀과 합성 → preview(GT)
        preview_from_gt = self._preview_sample(
            num_steps=num_steps,
            start_latents=z0_lat.float(),
            pixels_for_keep=x_gt,
            m=m,
            shared_noise=shared_noise,
            start_idx_override=start_idx,
        )

        # 원스텝 진단
        with torch.no_grad():
            B = Xi_lat.shape[0]
            sigma_b = sigma0.expand(B)
            sigma4  = sigma0.view(1,1,1,1).expand(B,1,1,1).float()
            x_t = Mi * Xi_noisy0 + (1.0 - Mi) * (sigma0 * shared_noise)

            prompt_embeds, pooled = self._encode_prompts(B)
            model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer
            was_train = model.training
            model.eval()

            x_t_scaled = self._fm_scale(x_t, sigma_b)
            raw = model(
                hidden_states=x_t_scaled.to(self.dtype),
                timestep=sigma_b,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=True
            ).sample.float()

            pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
            _, x0_pred = self._raw_to_eps_x0(raw, x_t, sigma4, pred_type)

            x_hat = from_latent_sd3(self.vae, x0_pred.to(self.dtype))
            K3 = m.repeat(1, 3, 1, 1)
            one_step_img = torch.clamp((K3 * x_gt + (1.0 - K3) * x_hat + 1.0) * 0.5, 0, 1).cpu()


        _, _, Hh, WW = x_gt.shape
        Ww = WW // 2
        person        = x_gt[:, :, :, :Ww]
        garment       = x_gt[:, :, :, Ww:]
        mask_keep     = m[:, :, :, :Ww]
        mask_vis      = mask_keep.repeat(1, 3, 1, 1)
        masked_person = person * mask_keep

        def _cpu32(t): return t.detach().to('cpu', dtype=torch.float32, non_blocking=True).contiguous()

        tiles_and_names = [
            (_cpu32(self._denorm(person))         , "person"),
            (_cpu32(mask_vis)                     , "mask_vis"),
            (_cpu32(self._denorm(masked_person))  , "masked_person"),
            (_cpu32(self._denorm(garment))        , "garment"),
            (xi_noisy_img                         , "Xi + noise"),
            (_cpu32(self._denorm(x_gt))           , "GT pair"),
            (preview_from_input                   , "preview(in)"),
            (preview_from_gt                      , "preview(GT)"),
            (one_step_img                         , "1-step TF"),
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

        with torch.no_grad():
            hole = (1.0 - m).to(preview_from_input)          # [B,1,H,2W]
            hole3 = hole.repeat(1,3,1,1)

            # 1) 출력 민감도: 두 경로 출력 차이(같은 노이즈, 다른 start_latents) - HOLE만
            diff_out = ((preview_from_input - preview_from_gt) * hole3).pow(2).sum(dim=(1,2,3))
            denom_out = hole3.sum(dim=(1,2,3)).clamp_min(1.0)
            mse_out_hole = (diff_out / denom_out).mean()

            # 2) 초기(스텝0) KEEP영역에서의 시작 차이 크기
            #   Xi_noisy0와 z0_noisy0를 같은 shared_noise로 만든 뒤 KEEP에서 차이 측정
            Xi_noisy0 = sched_probe.scale_noise(Xi_lat.float(), sigma0[None], shared_noise).float()
            z0_noisy0 = sched_probe.scale_noise(z0_lat.float(), sigma0[None], shared_noise).float()
            keep_lat_mask = Mi  # latent 해상도의 KEEP
            init_keep_delta = ((keep_lat_mask * (Xi_noisy0 - z0_noisy0)).pow(2)).mean()

            # 3) 출력 민감도 / 입력차 분모 (rough sensitivity)
            sens = (mse_out_hole / (init_keep_delta + 1e-8)).item()

            # 로그/TF
            self.tb.add_scalar("debug/preview_hole_mse", float(mse_out_hole), global_step)
            self.tb.add_scalar("debug/init_keep_delta",  float(init_keep_delta), global_step)
            self.tb.add_scalar("debug/output_sensitivity", sens, global_step)
            if hasattr(self, "logger"):
                self.logger.info(f"[probe] hole_mse={float(mse_out_hole):.5f} | "
                                f"init_keep_delta={float(init_keep_delta):.5f} | sens={sens:.3f}")

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
    def step(self, batch, global_step: int):
        H, W = self.cfg.size_h, self.cfg.size_w

        # 입/정답/마스크 모두 사용
        x_concat_in = batch["x_concat_in"].to(self.device, self.dtype)
        x_concat_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m_concat    = batch["m_concat"].to(self.device, self.dtype)

        with torch.no_grad():
            # VAE 인코딩
            z0 = self._encode_vae_latents(x_concat_gt, generator=None, sample=True).to(self.dtype)  # target x0
            Xi = self._encode_vae_latents(x_concat_in, generator=None, sample=True).to(self.dtype)  # inpaint input

            # KEEP 마스크를 latent 해상도로 (Mi=1 KEEP, Mi=0 HOLE)
            Mi = F.interpolate(
                m_concat,
                size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor),
                mode="nearest",
            ).to(self.dtype)

        bsz = z0.shape[0]
        noise = torch.randn_like(z0, dtype=torch.float32)

        # SD3 공식: u→indices→timesteps
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        num_tt = int(self.train_scheduler.timesteps.numel() - 1)
        indices = (u * num_tt).long().clamp_(0, num_tt - 1)
        timesteps = self.train_scheduler.timesteps[indices].to(device=z0.device, dtype=torch.float32)
        sigma_1d  = self.train_scheduler.sigmas[indices].to(device=z0.device, dtype=torch.float32)
        sigma4    = sigma_1d.view(-1, 1, 1, 1)  # [B,1,1,1]

        if self.is_main and global_step == 0:
            print("sched(sigmas)[:3] =", self.train_scheduler.sigmas[:3].float().cpu().tolist())
            print("sigma_1d[:3]      =", sigma_1d[:3].float().cpu().tolist())
            print("prediction_type =", self.train_scheduler.config.prediction_type)            
            self.logger.info(f"sched(sigmas)[:3] = {self.train_scheduler.sigmas[:3].float().cpu().tolist()}")
            self.logger.info(f"fsigma_1d[:3]     = {sigma_1d[:3].float().cpu().tolist()}")
            self.logger.info(f"prediction_type   = {self.train_scheduler.config.prediction_type}")


        # 둘 다 같은 noise로 re-noise
        z0_noisy = self.train_scheduler.scale_noise(z0.float(), sigma_1d, noise.float()).float()
        Xi_noisy = self.train_scheduler.scale_noise(Xi.float(), sigma_1d, noise.float()).float()

        # **진짜 학습 입력**: inpaint 분포와 동일하게 섞기 (KEEP=Mi)
        # noisy_model_input = Mi * Xi_noisy + (1.0 - Mi) * z0_noisy
        
        # inference와 동일하게 섞기
        sigma4 = sigma_1d.view(-1, 1, 1, 1)  # [B,1,1,1]
        noisy_model_input = Mi * Xi_noisy + (1.0 - Mi) * (sigma4 * noise.float())

        # FM 입력 스케일링        
        hs = self._fm_scale(noisy_model_input, sigma_1d)

        # 프롬프트 임베딩
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(bsz)

        # 모델 추론
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
            raw_out = self.transformer(
                hidden_states=hs.to(self.dtype),
                timestep=timesteps,  # FM에서는 sigma를 바로 timestep으로 사용
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0].float()

        # raw → (ε, x0) 변환할 때 **z_t는 방금 모델에 넣은 noisy_model_input**을 사용해야 함
        pred_type = str(getattr(self.train_scheduler.config, "prediction_type", "epsilon")).lower()
        _, x0_pred = self._raw_to_eps_x0(raw_out, noisy_model_input.float(), sigma4, pred_type)


        if getattr(self.cfg, "loss_sigma_weight", False):
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.cfg.weighting_scheme, sigmas=sigma_1d)
        else:
            weighting = torch.ones_like(sigma_1d)

        # 타깃은 GT의 x0 (z0), 손실은 HOLE만
        target = z0.float()
        Hmask = (1.0 - Mi).float()  # HOLE-only
        C = target.shape[1]

        diff2 = (x0_pred - target) ** 2
        per_elem = weighting.view(-1,1,1,1).float() * diff2
        masked = per_elem * Hmask

        B = masked.shape[0]
        masked_flat = masked.view(B, -1).sum(dim=1)
        denom_flat = (Hmask.view(B, -1).sum(dim=1) * C).clamp_min(1.0)
        loss_per_sample = masked_flat / denom_flat
        loss = loss_per_sample.mean()

        dbg = {
            "sigma_min": float(sigma_1d.min()),
            "sigma_max": float(sigma_1d.max()),
            "w_mean": float(weighting.mean()),
            "unweighted_mse": float(((diff2 * Hmask).view(B,-1).sum(dim=1) / denom_flat).mean()),
            "var_x0": float(target.pow(2).mean()),
            "hole_ratio": float(Mi[:, :, :, : Mi.shape[-1] // 2].float().mean()),
        }

        if (not torch.isfinite(loss)) or (not loss.requires_grad):
            return None
        return loss, dbg



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

        while global_step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            dbg_acc = {
                "sigma_min": 0.0, "sigma_max": 0.0, "w_mean": 0.0,
                "unweighted_mse": 0.0, "var_x0": 0.0, "hole_ratio": 0.0
            }
            nonfinite = False
            reason = ""
          

            for _ in range(self.cfg.grad_accum):
                batch = next(data_iter)
                out = self.step(batch, global_step)
                if out is None:
                    nonfinite = True; reason = "None/NaN in forward"; break
                loss, dbg = out

                if (not torch.isfinite(loss)) or (not loss.requires_grad):
                    nonfinite = True; reason = "non-finite/detached loss"; break

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += float(loss.detach().cpu())
                for k in dbg_acc.keys():
                    dbg_acc[k] += float(dbg[k])            

            if nonfinite:
                msg = f"[warn] skipping step {global_step}: {reason}."
                if self.is_main: pbar.write(msg)
                self.logger.info(msg)
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_scaler: self.scaler.update()
                global_step += 1
                if self.is_main: pbar.update(1)
                continue

            tf_grad_norm = self._grad_norm_sum(self.transformer)

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

            self._epoch_loss_sum += loss_accum / max(1, self.cfg.grad_accum)
            self._epoch_loss_count += 1
            train_loss_avg = self._epoch_loss_sum / max(1, self._epoch_loss_count)

            denom = float(max(1, self.cfg.grad_accum))
            dbg_avg = {k: v / denom for k, v in dbg_acc.items()}

            if self.is_main and ((global_step % self.cfg.log_every) == 0 or global_step == 1):
                self.tb.add_scalar("train/loss_total", train_loss_avg, global_step)
                self.tb.add_scalar("train/grad_norm_transformer", tf_grad_norm, global_step)

                prog = (global_step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0.0
                pct = int(prog * 100)
                line = (
                    f"Epoch {epoch}: {pct:3d}% | step {global_step}/{self.cfg.max_steps} "
                    f"| loss={train_loss_avg:.4f} "
                    f"| sigma[min,max]=[{dbg_avg['sigma_min']:.3f},{dbg_avg['sigma_max']:.3f}] "
                    f"| w_mean={dbg_avg['w_mean']:.4f} "
                    f"| unw_mse={dbg_avg['unweighted_mse']:.4f} "
                    f"| var_x0={dbg_avg['var_x0']:.4f} "
                    f"| hole={dbg_avg['hole_ratio']:.3f}"
                )
                pbar.set_postfix_str(f"loss={train_loss_avg:.4f}")
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

    "prefer_xformers": True,
    "disable_text": False,
    "save_root_dir": "logs", "save_name": "catvton_sd35_inpaint",
    "log_every": 50, "image_every": 500,
    "save_epoch_ckpt": 15,
    "default_resume_ckpt": None,
    "resume_ckpt": None,

    # Official SD3 training knobs
    "weighting_scheme": "logit_normal",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,

    # preview
    "preview_infer_steps": 60,
    "preview_seed": 1234,
    "preview_strength": 0.3,
    "preview_rows" : 4,
    "preview_caption_h": 28,

    # 추가: 고정 프롬프트
    "fixed_prompt": None,          # None면 기본 문구 사용    
    "disable_dynamic_shifting": True,  # 2W concat 입력에서 기본적으로 끔

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

    p.add_argument("--weighting_scheme", type=str, default=None, choices=["sigma_sqrt","logit_normal","mode","cosmap"])
    p.add_argument("--logit_mean", type=float, default=None)
    p.add_argument("--logit_std", type=float, default=None)
    p.add_argument("--mode_scale", type=float, default=None)
    
    
    # 고정 프롬프트
    p.add_argument("--fixed_prompt", type=str, default=None)
    p.add_argument("--disable_dynamic_shifting", action="store_true")

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

    # CLI 병합
    if getattr(args, "fixed_prompt", None) is not None:
        cfg["fixed_prompt"] = args.fixed_prompt    

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


# 수정후
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

    # rank 0에서 현재 실행 스크립트 백업
    if rank == 0:
        _try_copy_running_script(run_dirs["run_dir"])

    cfg_yaml_to_save = dict(cfg) if rank == 0 else None

    trainer = CatVTON_SD3_Trainer(cfg, run_dirs, cfg_yaml_to_save=cfg_yaml_to_save)
    trainer.train()

    if is_dist:
        dist.destroy_process_group()



if __name__ == "__main__":
    main()
