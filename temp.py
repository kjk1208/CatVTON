#!/usr/bin/env python3

# fixed_train_sd35_v2.py — SD3.5 CatVTON (Further Fixed version)
# 주요 수정사항:
# 1. 노이즈 스케일링 문제 해결
# 2. 프리뷰 생성 로직 개선
# 3. 마스크 처리 일관성 강화
# 4. VAE 스케일링 팩터 정확히 적용
# 5. 학습-추론 일관성 보장

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
    from diffusers import StableDiffusion3Img2ImgPipeline as StableDiffusion3Pipeline  # VTON은 Img2Img가 더 적합
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
# Utils
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
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        from datetime import timedelta
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
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

def _try_copy_running_script(dst_dir: str):
    import shutil
    try:
        script_path = os.path.abspath(sys.argv[0])
        if os.path.isfile(script_path):
            base = os.path.basename(script_path)
            dst_path = os.path.join(dst_dir, base)
            shutil.copy2(script_path, dst_path)
    except Exception as e:
        print(f"[warn] failed to copy running script: {e}")

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

        x_concat_in = torch.cat([x_p_in, x_g], dim=2)
        x_concat_gt = torch.cat([x_p,   x_g], dim=2)
        m_concat    = torch.cat([m, torch.ones_like(m)], dim=2)

        return {"x_concat_in": x_concat_in, "x_concat_gt": x_concat_gt, "m_concat": m_concat}

# ------------------------------------------------------------
# SD3.5 helpers (Fixed Version 2)
# ------------------------------------------------------------
@torch.no_grad()
def to_latent_sd3(vae, x, sample=True, generator=None):
    """수정된 VAE 인코딩 - 스케일링 팩터 정확히 적용"""
    vdtype = next(vae.parameters()).dtype
    posterior = vae.encode(x.to(vdtype)).latent_dist
    if sample:
        latents = posterior.sample(generator=generator)
    else:
        latents = posterior.mode()
    
    # SD3.5 정확한 스케일링 적용
    scaling_factor = getattr(vae.config, 'scaling_factor', 1.5305)
    shift_factor = getattr(vae.config, 'shift_factor', 0.0609)
    
    return (latents - shift_factor) * scaling_factor

@torch.no_grad()
def from_latent_sd3(vae, z: torch.Tensor) -> torch.Tensor:
    """수정된 VAE 디코딩 - 스케일링 팩터 정확히 역적용"""
    vdtype = next(vae.parameters()).dtype
    z = z.to(vdtype)
    
    # SD3.5 정확한 스케일링 역적용
    scaling_factor = getattr(vae.config, 'scaling_factor', 1.5305)
    shift_factor = getattr(vae.config, 'shift_factor', 0.0609)
    
    z = z / scaling_factor + shift_factor
    img = vae.decode(z).sample
    return img

# ------------------------------------------------------------
# Enhanced FlowMatch Scheduler
# ------------------------------------------------------------
def create_enhanced_fm_scheduler(base_scheduler_config, device, height, width, transformer):
    """향상된 FlowMatch 스케줄러 생성 (더 안정적인 노이즈 스케줄링)"""
    config = dict(base_scheduler_config)
    
    # SD3.5 안정화 설정
    config.update({
        "num_train_timesteps": 1000,
        "shift": 3.0,  # SD3.5 기본값 (더 부드러운 노이즈 스케줄)
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096
    })
    
    scheduler = FM.from_config(config)
    
    # 정확한 VAE 스케일 팩터
    vae_scale_factor = 8
    patch_size = getattr(transformer.config, "patch_size", 2)
    
    # 이미지 토큰 길이 계산 (2W 때문에 가로가 2배)
    h_tokens = height // vae_scale_factor // patch_size
    w_tokens = (2 * width) // vae_scale_factor // patch_size
    image_seq_len = h_tokens * w_tokens
    
    # Dynamic shifting 계산
    base_len = config["base_image_seq_len"]
    max_len = config["max_image_seq_len"]
    base_s = config["base_shift"]
    max_s = config["max_shift"]
    
    mu = base_s + (max_s - base_s) * max(0.0, (image_seq_len - base_len) / max(1.0, (max_len - base_len)))
    mu = max(base_s, min(max_s, mu))
    
    return scheduler, mu

def setup_enhanced_timesteps(scheduler, num_steps, device, mu=None):
    """개선된 timesteps 설정"""
    try:
        if mu is not None:
            scheduler.set_timesteps(num_steps, device=device, mu=float(mu))
        else:
            scheduler.set_timesteps(num_steps, device=device)
    except (TypeError, AttributeError):
        # mu를 지원하지 않는 경우 기본 설정 사용
        scheduler.set_timesteps(num_steps, device=device)
    
    # timesteps 정규화 및 검증
    timesteps = scheduler.timesteps.clone()
    
    # 내림차순 정렬 확인
    if len(timesteps) > 1 and timesteps[0] < timesteps[1]:
        timesteps = torch.flip(timesteps, [0])
    
    # 마지막이 0에 가깝지 않으면 추가
    if timesteps[-1] > 1e-7:
        timesteps = torch.cat([timesteps, timesteps.new_zeros(1)])
    
    scheduler.timesteps = timesteps
    return scheduler, timesteps

# ------------------------------------------------------------
# Freezing
# ------------------------------------------------------------
def freeze_all_but_self_attn_qkv(transformer, open_to_add_out: bool=False, open_io: bool=False, 
                                 open_ffn_blocks: list=None, open_block_norms: bool=False):
    # 모든 파라미터 동결
    for p in transformer.parameters():
        p.requires_grad = False

    # Self-attention Q/K/V/Out만 학습 가능하게
    for name, m in transformer.named_modules():
        if name.endswith(".attn") or hasattr(m, "to_q"):
            for attr in ["to_q", "to_k", "to_v", "to_out"]:
                sub = getattr(m, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad = True

    # Cross-attention은 동결 유지
    for name, m in transformer.named_modules():
        if name.endswith(".attn"):
            for attr in ["add_k_proj", "add_v_proj", "add_q_proj"]:
                sub = getattr(m, attr, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad = False
            
            sub = getattr(m, "to_add_out", None)
            if sub is not None:
                for p in sub.parameters():
                    p.requires_grad = bool(open_to_add_out)

    # I/O 어댑터
    if open_io:
        if hasattr(transformer, "pos_embed") and hasattr(transformer.pos_embed, "proj"):
            for p in transformer.pos_embed.proj.parameters():
                p.requires_grad = True
        for p in transformer.norm_out.parameters():
            p.requires_grad = True
        for p in transformer.proj_out.parameters():
            p.requires_grad = True

    # FFN 블록
    if open_ffn_blocks is not None:
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is not None:
            for idx in open_ffn_blocks:
                if 0 <= idx < len(blocks):
                    blk = blocks[idx]
                    for attr in ["ff", "ff_context"]:
                        sub = getattr(blk, attr, None)
                        if sub is not None:
                            for p in sub.parameters(): 
                                p.requires_grad = True
                    if open_block_norms:
                        for attr in ["norm2", "norm2_context"]:
                            sub = getattr(blk, attr, None)
                            if sub is not None:
                                for p in sub.parameters(): 
                                    p.requires_grad = True

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No params unfrozen; check patterns.")
    
    return trainable

# ------------------------------------------------------------
# Enhanced Trainer
# ------------------------------------------------------------
class EnhancedCatVTON_SD3_Trainer:
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
        self.dtype = mp2dtype.get(cfg.mixed_precision, torch.bfloat16)

        # 디렉토리 및 로깅 설정
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

        # 설정 저장
        if cfg_yaml_to_save is not None and yaml is not None and self.is_main:
            with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg_yaml_to_save, f, sort_keys=False)

        # HF 토큰 처리
        env_token = (os.environ.get("HUGGINGFACE_TOKEN")
                     or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                     or os.environ.get("HF_TOKEN"))
        token = cfg.hf_token or env_token

        # SD3.5 모델 로드
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

        # 파이프라인 로드
        pipe = StableDiffusion3Pipeline.from_pretrained(
            local_dir, torch_dtype=self.dtype, local_files_only=True, use_safetensors=True,
        ).to(self.device)

        self.vae = pipe.vae
        self.transformer: SD3Transformer2DModel = pipe.transformer
        self.encode_prompt = pipe.encode_prompt
        
        # VAE 스케일링 팩터 확인
        self.vae_scale_factor = 8
        if self.is_main:
            sf = getattr(self.vae.config, 'scaling_factor', 1.5305)
            shift = getattr(self.vae.config, 'shift_factor', 0.0609)
            self.logger.info(f"VAE scaling_factor={sf}, shift_factor={shift}")
            
            # 디버깅: VAE 설정 전체 출력
            self.logger.info(f"VAE config: {self.vae.config}")
            
            # 디버깅: 테스트 인코딩/디코딩 체크
            test_img = torch.randn(1, 3, 512, 768, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                test_lat = to_latent_sd3(self.vae, test_img)
                test_rec = from_latent_sd3(self.vae, test_lat)
                self.logger.info(f"Test encode/decode - input range: [{test_img.min():.3f}, {test_img.max():.3f}]")
                self.logger.info(f"Test encode/decode - latent range: [{test_lat.min():.3f}, {test_lat.max():.3f}]") 
                self.logger.info(f"Test encode/decode - output range: [{test_rec.min():.3f}, {test_rec.max():.3f}]")

        # 향상된 스케줄러 생성
        self.train_scheduler, self.train_mu = create_enhanced_fm_scheduler(
            pipe.scheduler.config, self.device, cfg.size_h, cfg.size_w, self.transformer
        )
        self.infer_scheduler, self.infer_mu = create_enhanced_fm_scheduler(
            pipe.scheduler.config, self.device, cfg.size_h, cfg.size_w, self.transformer
        )

        if self.is_main:
            self.logger.info(f"Enhanced FlowMatch scheduler with mu_train={self.train_mu:.3f}")
            self.logger.info(f"Loaded SD3.5 model: {cfg.sd3_model}")

        # 메모리 최적화
        try:
            self.transformer.enable_gradient_checkpointing()
            if self.is_main: 
                print("[mem] gradient checkpointing ON")
        except Exception as e:
            if self.is_main: 
                print(f"[mem] gradient checkpointing not available: {e}")

        # 파라미터 동결
        tf = self.transformer
        N = len(tf.transformer_blocks)
        mid_ffn_blocks = list(range(N//3, min(N, 2*N//3 + 2)))

        trainable_tf = freeze_all_but_self_attn_qkv(
            transformer=self.transformer, 
            open_io=True, 
            open_ffn_blocks=mid_ffn_blocks, 
            open_block_norms=False
        )
        
        # 안정성을 위해 trainable 파라미터를 fp32로 유지
        if self.dtype in [torch.float16, torch.bfloat16]:
            for _, p in self.transformer.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

        if self.is_main:
            print(f"Trainable params (transformer only)={trainable_tf/1e6:.2f}M")
            _debug_print_trainables(self.transformer, "after_freeze_before_ddp")

        # 데이터셋
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
            num_workers=cfg.num_workers, 
            pin_memory=True, 
            drop_last=True
        )
        
        self.logger.info(f"Dataset len={len(self.dataset)} batch_size(per-rank)={cfg.batch_size}")
        self.logger.info("Enhanced mask semantics: Mi=1 KEEP, Mi=0 HOLE")

        if self.is_dist:
            self.steps_per_epoch = max(1, self.sampler.num_samples // cfg.batch_size)
        else:
            self.steps_per_epoch = max(1, len(self.dataset) // cfg.batch_size)

        # AMP 스케일러
        self.use_scaler = (self.device.type == "cuda") and (self.dtype == torch.float16)
        if self.is_main:
            print(f"[amp] dtype={self.dtype}, use_scaler={self.use_scaler}")

        # DDP 래핑
        self.transformer = DDP(
            self.transformer, 
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
            broadcast_buffers=False, 
            find_unused_parameters=False
        )

        # 향상된 옵티마이저 설정
        self._setup_enhanced_optimizer()

        # 체크포인트 복원
        self.start_epoch = 0
        self.start_step = 0
        resume_path = cfg.resume_ckpt or cfg.default_resume_ckpt
        self._resume_from_ckpt_if_needed(resume_path)

        # 초기화
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0
        self._preview_gen = torch.Generator(device=self.device).manual_seed(cfg.preview_seed)
        self._mask_keep_logged = False

        # 스케일러
        self.scaler = torch_amp.GradScaler(enabled=self.use_scaler)

    def _setup_enhanced_optimizer(self):
        """향상된 옵티마이저 설정 - 파라미터 그룹별 차등 학습률"""
        qkv_params, io_params, ffn_params, other_params = [], [], [], []
        
        for n, p in self.transformer.named_parameters():
            if not p.requires_grad:
                continue
            
            if any(x in n for x in [".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out"]):
                qkv_params.append(p)
            elif any(x in n for x in ["pos_embed.proj", "norm_out", "proj_out"]):
                io_params.append(p)
            elif any(x in n for x in [".ff.", ".ff_context."]):
                ffn_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        base_lr = self.cfg.lr
        
        if qkv_params:
            param_groups.append({"params": qkv_params, "lr": base_lr, "name": "qkv"})
        if io_params:
            param_groups.append({"params": io_params, "lr": base_lr * 1.5, "name": "io"})
        if ffn_params:
            param_groups.append({"params": ffn_params, "lr": base_lr * 0.8, "name": "ffn"})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})
        
        if not param_groups:
            tf_params = [p for p in self.transformer.parameters() if p.requires_grad]
            param_groups = [{"params": tf_params, "lr": base_lr, "name": "all"}]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),  # 더 안정적인 베타값
            eps=1e-8, 
            weight_decay=1e-4  # 약간의 정규화
        )

    def _encode_prompts(self, bsz: int):
        """개선된 프롬프트 인코딩"""
        if getattr(self.cfg, "disable_text", False):
            if not hasattr(self, "_uncond_pe"):
                pe, _, ppe, _ = self.encode_prompt(
                    prompt="", prompt_2="", prompt_3="",
                    device=self.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                self._uncond_pe = pe.detach().to(self.dtype)
                self._uncond_ppe = ppe.detach().to(self.dtype)

            return (self._uncond_pe.expand(bsz, -1, -1).contiguous(),
                    self._uncond_ppe.expand(bsz, -1).contiguous())

        # 더 구체적인 프롬프트
        fixed_txt = getattr(
            self.cfg,
            "fixed_prompt",
            "high quality, photorealistic, detailed clothing texture, natural lighting, sharp focus"
        )
        
        if not hasattr(self, "_fixed_pe"):
            pe, _, ppe, _ = self.encode_prompt(
                prompt=fixed_txt,
                prompt_2=fixed_txt,
                prompt_3=fixed_txt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            self._fixed_pe = pe.detach().to(self.dtype)
            self._fixed_ppe = ppe.detach().to(self.dtype)

        return (self._fixed_pe.expand(bsz, -1, -1).contiguous(),
                self._fixed_ppe.expand(bsz, -1).contiguous())

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

    def _enhanced_noise_schedule(self, x: torch.Tensor, timestep) -> torch.Tensor:
        """향상된 노이즈 스케일링 - 더 안정적인 방식"""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=x.device, dtype=torch.float32)
        
        timestep = timestep.to(device=x.device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep[None]
        
        # SD3.5 권장 노이즈 스케일링
        sigma = timestep.view(-1, *([1] * (x.ndim - 1)))
        
        # FlowMatch의 정확한 스케일링 공식
        x_f32 = x.float()
        alpha_t = 1.0 / torch.sqrt(1.0 + sigma * sigma)
        return (x_f32 * alpha_t).to(x.dtype)

    def _consistent_add_noise(self, clean_latents, noise, timestep):
        """일관된 노이즈 추가 방식 - 더 정확한 공식"""
        timestep = torch.as_tensor(timestep, device=clean_latents.device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep.view(1)
        if timestep.shape[0] != clean_latents.shape[0]:
            timestep = timestep[:1].repeat(clean_latents.shape[0])
        
        # SD3.5 FlowMatch 공식: x_t = (1-t) * x_0 + t * noise
        # 여기서 t는 정규화된 timestep (0~1 범위)
        t_normalized = timestep.view(-1, 1, 1, 1) / 1000.0  # 1000은 max timesteps
        t_normalized = torch.clamp(t_normalized, 0.0, 1.0)
        
        return ((1.0 - t_normalized) * clean_latents.float() + 
                t_normalized * noise.float()).to(clean_latents.dtype)

    @torch.no_grad()
    def _enhanced_preview_sample(self, num_steps: int, start_latents: torch.Tensor, 
                                pixels_for_keep: torch.Tensor, m: torch.Tensor,
                                shared_noise: Optional[torch.Tensor] = None,
                                start_idx_override: Optional[int] = None):
        """향상된 프리뷰 샘플링 - 더 안정적인 인페인팅"""
        H, W = self.cfg.size_h, self.cfg.size_w
        device = self.device
        dtype = self.dtype

        # 마스크를 latent 해상도로 변환
        Mi = F.interpolate(
            m.to(device, dtype),
            size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor),
            mode="nearest"
        ).float()
        
        B = start_latents.shape[0]

        # 스케줄러 설정
        scheduler, timesteps = setup_enhanced_timesteps(
            self.infer_scheduler, num_steps, device, self.infer_mu
        )

        # 시작 인덱스 결정
        if start_idx_override is None:
            s = max(0.0, min(1.0, float(self.cfg.preview_strength)))
            N = len(timesteps) - 1
            start_idx = min(max(int((1.0 - s) * N), 0), N - 1)
        else:
            start_idx = int(start_idx_override)

        timesteps_sub = timesteps[start_idx:].contiguous()
        if timesteps_sub[-1] > 1e-7:
            timesteps_sub = torch.cat([timesteps_sub, timesteps_sub.new_zeros(1)])

        # 공유 노이즈 생성
        if shared_noise is None:
            shared_noise = torch.randn(start_latents.shape, dtype=start_latents.dtype, 
                                          device=device, generator=self._preview_gen)

        # 초기 노이즈 상태
        t0 = timesteps_sub[0]
        start_noisy = self._consistent_add_noise(start_latents.float(), shared_noise, t0)
        
        # 인페인팅 초기화: KEEP 영역은 원본, HOLE 영역은 노이즈
        z = Mi * start_noisy + (1.0 - Mi) * shared_noise * (t0 / 1000.0)

        # 프롬프트 임베딩
        prompt_embeds, pooled = self._encode_prompts(B)
        model = self.transformer.module if isinstance(self.transformer, DDP) else self.transformer

        was_train = model.training
        model.eval()
        
        try:
            for i in range(len(timesteps_sub) - 1):
                t_curr = timesteps_sub[i].to(device, torch.float32)
                t_next = timesteps_sub[i + 1].to(device, torch.float32)
                
                t_batch = t_curr.expand(B).to(device)
                
                # 모델 입력 스케일링
                z_scaled = self._enhanced_noise_schedule(z, t_batch)
                
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device.type=='cuda')):
                    model_output = model(
                        hidden_states=z_scaled.to(dtype),
                        timestep=t_batch,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=True,
                    ).sample.float()

                # Euler 스텝
                dt = t_next - t_curr
                z_next = z + dt * model_output
                
                # 인페인팅 재합성
                if t_next > 1e-7:
                    init_latents_proper = self._consistent_add_noise(start_latents.float(), shared_noise, t_next)
                    z = Mi * init_latents_proper + (1.0 - Mi) * z_next
                else:
                    z = Mi * start_latents.float() + (1.0 - Mi) * z_next

            # VAE 디코드 및 최종 합성
            x_hat = from_latent_sd3(self.vae, z.to(dtype))
            
            # 학습 초기에는 더 관대한 clamp (자연스러운 색상 분포)
            # 하지만 여전히 극단적인 saturation은 방지
            x_hat = torch.clamp(x_hat, -1.2, 1.2)
            
            K3_keep = m.to(device, dtype).repeat(1, 3, 1, 1)
            x_final = K3_keep * pixels_for_keep.to(device, dtype) + (1.0 - K3_keep) * x_hat
            
            # 디버깅: saturation 체크
            if self.is_main:
                self.logger.info(f"Preview debug - x_hat range (after clamp): [{x_hat.min():.3f}, {x_hat.max():.3f}]")
                self.logger.info(f"Preview debug - x_final range: [{x_final.min():.3f}, {x_final.max():.3f}]")
                self.logger.info(f"Preview debug - pixels_for_keep range: [{pixels_for_keep.min():.3f}, {pixels_for_keep.max():.3f}]")
            
            return torch.clamp((x_final + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
            
        finally:
            if was_train:
                model.train()
            
            # 프리뷰 샘플링 중간 텐서들 정리
            if 'z' in locals():
                del z
            if 'z_scaled' in locals():
                del z_scaled
            if 'model_output' in locals():
                del model_output
            if 'z_next' in locals():
                del z_next
            if 'x_hat' in locals():
                del x_hat
            if 'x_final' in locals():
                del x_final
                
    def _calculate_mu_for_resolution(self, height, width, transformer):
        """SD3.5의 정확한 mu 계산 방식"""
        # SD3.5 dynamic shifting 파라미터들
        base_shift = 0.5
        max_shift = 1.15
        base_image_seq_len = 256
        max_image_seq_len = 4096
        
        # VAE 스케일링과 패치 크기 고려
        vae_scale_factor = 8
        patch_size = getattr(transformer.config, "patch_size", 2)
        
        # 이미지 토큰 길이 계산 (concat된 이미지이므로 2W)
        h_tokens = height // vae_scale_factor // patch_size
        w_tokens = (2 * width) // vae_scale_factor // patch_size
        image_seq_len = h_tokens * w_tokens
        
        # Dynamic shifting 계산 공식
        mu = base_shift + (max_shift - base_shift) * max(0.0, 
            (image_seq_len - base_image_seq_len) / max(1.0, (max_image_seq_len - base_image_seq_len)))
        mu = max(base_shift, min(max_shift, mu))
        
        return mu

    @torch.no_grad()
    def _save_enhanced_preview(self, batch: Dict[str, torch.Tensor], global_step: int, max_rows: int = 4):
        """향상된 프리뷰 저장 - 디버깅 추가"""
        if not self.is_main:
            self.logger.debug(f"Skipping preview: not main process")
            return

        self.logger.info(f"Starting preview generation at step {global_step}")
        
        try:
            rows = min(int(self.cfg.get("preview_rows", 2)), max_rows)
            self.logger.info(f"Preview rows: {rows}, batch sizes: {[v.shape for v in batch.values()]}")
            
            x_in = batch["x_concat_in"][:rows].to(self.device, self.dtype)
            x_gt = batch["x_concat_gt"][:rows].to(self.device, self.dtype)
            m = batch["m_concat"][:rows].to(self.device, self.dtype)
            
            self.logger.info(f"Tensors moved to device. Shapes: x_in={x_in.shape}, x_gt={x_gt.shape}, m={m.shape}")
            
            if any(t.shape[0] == 0 for t in [x_in, x_gt, m]):
                self.logger.warning(f"Empty tensors detected, skipping preview")
                return

            # 마스크 통계 로깅 (한 번만)
            if not self._mask_keep_logged:
                keep_ratio = m[:, :, :, : m.shape[-1] // 2].float().mean().item()
                hole_ratio = 1.0 - keep_ratio
                self.logger.info(f"[mask] KEEP ratio={keep_ratio:.3f}, HOLE ratio={hole_ratio:.3f}")
                self._mask_keep_logged = True

            num_steps = max(4, int(self.cfg.preview_infer_steps))
            self.logger.info(f"Preview inference steps: {num_steps}")
            
            # VAE 인코딩
            self.logger.info("Starting VAE encoding...")
            with torch.no_grad():
                Xi_lat = to_latent_sd3(self.vae, x_in, sample=True, generator=self._preview_gen).float()
                z0_lat = to_latent_sd3(self.vae, x_gt, sample=True, generator=self._preview_gen).float()
            
            self.logger.info(f"VAE encoding done. Latent shapes: Xi_lat={Xi_lat.shape}, z0_lat={z0_lat.shape}")
            
            shared_noise = torch.randn(Xi_lat.shape, dtype=Xi_lat.dtype, 
                                          device=Xi_lat.device, generator=self._preview_gen)
            self.logger.info(f"Shared noise generated: {shared_noise.shape}")

            # 다양한 강도로 프리뷰 생성
            strength = float(self.cfg.preview_strength)
            self.logger.info(f"Generating preview from input with strength {strength}")
            
            preview_from_input = self._enhanced_preview_sample(
                num_steps=num_steps,
                start_latents=Xi_lat,
                pixels_for_keep=x_in,
                m=m,
                shared_noise=shared_noise,
            )
            self.logger.info(f"Preview from input generated: {preview_from_input.shape}")

            self.logger.info("Generating preview from GT")
            preview_from_gt = self._enhanced_preview_sample(
                num_steps=num_steps,
                start_latents=z0_lat,
                pixels_for_keep=x_gt,
                m=m,
                shared_noise=shared_noise,
            )
            self.logger.info(f"Preview from GT generated: {preview_from_gt.shape}")

            # 초기 노이즈 상태 시각화
            self.logger.info("Generating noise visualization")
            scheduler_vis = FM.from_config(self.infer_scheduler.config)
            scheduler_vis.set_timesteps(num_steps, device=device)
            timesteps = scheduler_vis.timesteps
            
            Mi_lat = F.interpolate(
                m, size=(self.cfg.size_h // self.vae_scale_factor, (2 * self.cfg.size_w) // self.vae_scale_factor),
                mode="nearest"
            ).float()
            
            Xi_noisy0 = self._consistent_add_noise(Xi_lat, shared_noise, t0)
            z_init_vis = Mi_lat * Xi_noisy0 + (1.0 - Mi_lat) * shared_noise * (t0 / 1000.0)
            xi_noisy_img = from_latent_sd3(self.vae, z_init_vis.to(self.dtype))
            
            # 노이즈 시각화도 관대한 clamp 적용
            xi_noisy_img = torch.clamp(xi_noisy_img, -1.2, 1.2)
            xi_noisy_img = torch.clamp((xi_noisy_img + 1.0) * 0.5, 0.0, 1.0).detach().cpu()
            self.logger.info(f"Noise visualization generated: {xi_noisy_img.shape}")

            # 시각화 구성요소 준비
            _, _, Hh, WW = x_gt.shape
            Ww = WW // 2
            person = x_gt[:, :, :, :Ww]
            garment = x_gt[:, :, :, Ww:]
            mask_keep = m[:, :, :, :Ww]
            mask_vis = mask_keep.repeat(1, 3, 1, 1)
            masked_person = person * mask_keep

            def _cpu32(t): 
                return t.detach().to('cpu', dtype=torch.float32, non_blocking=True).contiguous()

            # 타일 구성
            self.logger.info("Composing visualization tiles")
            tiles_and_names = [
                (_cpu32(self._denorm(person)), "person"),
                (_cpu32(mask_vis), "mask"),
                (_cpu32(self._denorm(masked_person)), "masked"),
                (_cpu32(self._denorm(garment)), "garment"),
                (xi_noisy_img, f"noisy(t={t0:.0f})"),
                (_cpu32(self._denorm(x_gt)), "GT"),
                (preview_from_input, "pred(input)"),
                (preview_from_gt, "pred(GT)"),
            ]

            # 그리드 생성
            self.logger.info("Creating image grid")
            cols = [t for (t, _) in tiles_and_names]
            panel_bchw = torch.cat(cols, dim=3)
            col_widths = [int(t.shape[-1]) for (t, _) in tiles_and_names]
            col_names = [name for (_, name) in tiles_and_names]

            grid = make_grid(panel_bchw, nrow=1, padding=0)
            
            # 캡션 추가
            self.logger.info("Adding captions")
            # 2) 캡션 생성
            caption_h = int(self.cfg.get("preview_caption_h", 32))
            bottom = self._bottom_caption_row(
                col_names, col_widths, height=caption_h, scale=2.0, bg=0.95, fg=0.1
            )[0]  # (3, Hc, Wb)

            # 3) 혹시 모를 폭 불일치를 안전하게 보정
            Wg = int(grid.shape[-1])
            Wb = int(bottom.shape[-1])
            if Wb != Wg:
                if Wb < Wg:
                    pad_left  = (Wg - Wb) // 2
                    pad_right = Wg - Wb - pad_left
                    bottom = F.pad(bottom.unsqueeze(0), (pad_left, pad_right, 0, 0)).squeeze(0)
                else:
                    bottom = bottom[:, :, :Wg]

            # 4) 세로 방향으로 붙임 (C, H_sum, W)
            final_img = torch.cat([grid, bottom], dim=1)

            # 저장
            self.logger.info("Saving final image")
            out_path = os.path.join(self.img_dir, f"step_{global_step:06d}.png")
            
            # 디렉토리 존재 확인
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            save_image(final_img, out_path)
            self.logger.info(f"✓ Enhanced preview saved successfully: {out_path}")
            
        except Exception as e:
            self.logger.error(f"✗ Preview generation failed at step {global_step}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # 디버깅을 위해 예외 재발생
        finally:
            # GPU 메모리 정리
            if 'Xi_lat' in locals():
                del Xi_lat
            if 'z0_lat' in locals():
                del z0_lat
            if 'shared_noise' in locals():
                del shared_noise
            if 'x_in' in locals():
                del x_in
            if 'x_gt' in locals():
                del x_gt
            if 'm' in locals():
                del m
            if 'person' in locals():
                del person, garment, mask_keep, mask_vis, masked_person
            
            # GPU 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("GPU memory cleaned after preview generation")

    @torch.no_grad()
    def _bottom_caption_row(self, names: List[str], widths: List[int],
                            height: int = 32, scale: float = 2.0,
                            bg: float = 0.95, fg: float = 0.1) -> torch.Tensor:
        """캡션 행 생성"""
        W_total = int(sum(map(int, widths)))
        H = int(height * scale)
        bg255 = int(bg * 255)
        fg255 = int(fg * 255)
        img = Image.new("RGB", (W_total, H), (bg255, bg255, bg255))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(H * 0.6))
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
            x = x0 + max(0, (w - tw) // 2)
            y = max(0, (H - th) // 2)
            draw.text((x, y), name, fill=(fg255, fg255, fg255), font=font)
            x0 += w

        t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)

    def enhanced_step(self, batch, global_step: int):
        """향상된 학습 스텝 - 더 안정적인 loss 계산"""
        H, W = self.cfg.size_h, self.cfg.size_w

        x_concat_in = batch["x_concat_in"].to(self.device, self.dtype)
        x_concat_gt = batch["x_concat_gt"].to(self.device, self.dtype)
        m_concat = batch["m_concat"].to(self.device, self.dtype)

        with torch.no_grad():
            # VAE 인코딩
            z0 = to_latent_sd3(self.vae, x_concat_gt, sample=True).float()  # target
            Xi = to_latent_sd3(self.vae, x_concat_in, sample=True).float()  # input

            # 마스크를 latent 해상도로 변환
            Mi = F.interpolate(
                m_concat,
                size=(H // self.vae_scale_factor, (2 * W) // self.vae_scale_factor),
                mode="nearest",
            ).float()

        bsz = z0.shape[0]
        noise = torch.randn_like(z0, dtype=torch.float32)

        # 향상된 timestep 샘플링
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        
        # 스케줄러에서 timesteps 가져오기
        scheduler, timesteps_all = setup_enhanced_timesteps(self.train_scheduler, 1000, self.device, self.train_mu)
        num_tt = len(timesteps_all) - 1
        indices = (u * num_tt).long().clamp_(0, num_tt - 1)
        timesteps = timesteps_all[indices].to(device=z0.device, dtype=torch.float32)

        # 로깅 (첫 스텝에만)
        if self.is_main and global_step <= 1:
            self.logger.info(f"Enhanced timesteps range: [{timesteps_all[0]:.1f}, {timesteps_all[-1]:.1f}]")
            self.logger.info(f"Sample timesteps: {timesteps[:3].cpu().tolist()}")

        # 일관된 노이즈 주입
        z0_noisy = self._consistent_add_noise(z0, noise, timesteps)
        Xi_noisy = self._consistent_add_noise(Xi, noise, timesteps)

        # 인페인팅 입력 구성
        t_norm = (timesteps.view(-1, 1, 1, 1) / 1000.0).clamp(0, 1)
        noisy_model_input = Mi * Xi_noisy + (1.0 - Mi) * (noise * t_norm)

        # 모델 입력 스케일링
        hs = self._enhanced_noise_schedule(noisy_model_input, timesteps)

        # 프롬프트 임베딩
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(bsz)

        # 모델 추론
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.device.type == 'cuda')):
            model_output = self.transformer(
                hidden_states=hs.to(self.dtype),
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0].float()

        # 예측값을 x0로 변환 (FlowMatch에서는 velocity prediction)
        # v = (z_1 - z_0), 따라서 z_0 = z_t - t * v
        t_norm_4d = t_norm
        x0_pred = noisy_model_input - t_norm_4d * model_output

        # 손실 계산 - HOLE 영역만 고려
        if getattr(self.cfg, "loss_sigma_weight", True):
            # timestep 기반 가중치
            weighting = 1.0 / (1.0 + timesteps / 100.0)  # 높은 timestep일수록 낮은 가중치
        else:
            weighting = torch.ones_like(timesteps)

        target = z0  # GT latent
        Hmask = (1.0 - Mi)  # HOLE mask (Mi=1 KEEP, Mi=0 HOLE)
        
        # L2 loss in HOLE region only
        diff = x0_pred - target
        diff_masked = diff * Hmask
        
        # 가중 평균
        B = diff_masked.shape[0]
        loss_per_sample = weighting.view(-1, 1, 1, 1) * (diff_masked ** 2)
        
        # 정규화 (HOLE 영역 크기로 나누기)
        hole_pixels = Hmask.view(B, -1).sum(dim=1).clamp_min(1.0)
        loss_normalized = loss_per_sample.view(B, -1).sum(dim=1) / hole_pixels
        loss = loss_normalized.mean()

        # 디버그 정보
        dbg = {
            "t_min": float(timesteps.min()),
            "t_max": float(timesteps.max()),
            "w_mean": float(weighting.mean()),
            "mse_raw": float(((diff * Hmask) ** 2).mean()),
            "target_var": float(target.var()),
            "pred_var": float(x0_pred.var()),
            "hole_ratio": float(Hmask.mean()),
        }

        if (not torch.isfinite(loss)) or (not loss.requires_grad):
            return None
        return loss, dbg

    def _save_ckpt(self, epoch: int, train_loss_epoch: float, global_step: int) -> str:
        ckpt_path = os.path.join(self.model_dir, f"enhanced_epoch_{epoch}_loss_{train_loss_epoch:.04f}.ckpt")
        payload = {
            "transformer": ddp_state_dict(self.transformer),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "cfg": dict(self.cfg),
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_epoch": float(train_loss_epoch),
        }
        if self.is_main:
            torch.save(payload, ckpt_path)
            self.logger.info(f"Enhanced checkpoint saved: {ckpt_path}")
        return ckpt_path

    def _resume_from_ckpt_if_needed(self, resume_path: Optional[str]):
        if not resume_path or not os.path.isfile(resume_path):
            return
        
        if self.is_main:
            self.logger.info(f"Resuming from checkpoint: {resume_path}")
        
        ckpt = torch.load(resume_path, map_location="cpu")
        
        try:
            missing, unexpected = self.transformer.load_state_dict(ckpt["transformer"], strict=False)
            if self.is_main and (missing or unexpected):
                self.logger.info(f"Resume - missing: {len(missing)}, unexpected: {len(unexpected)}")
        except Exception as e:
            if self.is_main:
                self.logger.warning(f"Failed to load transformer state: {e}")

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.start_step = int(ckpt.get("global_step", 0))
        
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                # GPU로 옮기기
                device = next(self.transformer.parameters()).device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            except Exception as e:
                if self.is_main:
                    self.logger.warning(f"Failed to load optimizer state: {e}")
        
        if "scaler" in ckpt and self.use_scaler:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if self.is_main:
                    self.logger.warning(f"Failed to load scaler state: {e}")

        if self.is_main:
            self.logger.info(f"Resumed from epoch {self.start_epoch}, step {self.start_step}")

    def train(self):
        """향상된 학습 루프"""
        global_step = getattr(self, "start_step", 0)
        epoch = getattr(self, "start_epoch", 0)
        self.transformer.train()

        data_iter = itertools.cycle(self.loader)
        pbar = tqdm(
            total=self.cfg.max_steps,
            dynamic_ncols=True,
            desc=f"Enhanced Epoch {epoch}",
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
                "t_min": 0.0, "t_max": 0.0, "w_mean": 0.0,
                "mse_raw": 0.0, "target_var": 0.0, "pred_var": 0.0, "hole_ratio": 0.0
            }
            
            valid_steps = 0

            # Gradient accumulation
            for micro_step in range(self.cfg.grad_accum):
                batch = next(data_iter)
                out = self.enhanced_step(batch, global_step)
                
                if out is None:
                    if self.is_main and global_step % 100 == 0:
                        self.logger.warning(f"Step {global_step}: invalid loss in micro_step {micro_step}")
                    continue

                loss, dbg = out
                
                if (not torch.isfinite(loss)) or (not loss.requires_grad):
                    if self.is_main and global_step % 100 == 0:
                        self.logger.warning(f"Step {global_step}: non-finite loss={loss}")
                    continue

                # Normalize loss by grad_accum
                loss = loss / self.cfg.grad_accum
                
                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += float(loss.detach().cpu()) * self.cfg.grad_accum
                for k in dbg_acc.keys():
                    dbg_acc[k] += float(dbg[k])
                valid_steps += 1

            if valid_steps == 0:
                if self.is_main:
                    self.logger.warning(f"Step {global_step}: no valid micro-steps")
                global_step += 1
                if self.is_main: 
                    pbar.update(1)
                continue

            # Gradient clipping and optimizer step
            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            if self.use_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            global_step += 1
            if self.is_main: 
                pbar.update(1)

            # Loss tracking
            self._epoch_loss_sum += loss_accum
            self._epoch_loss_count += 1
            train_loss_avg = self._epoch_loss_sum / max(1, self._epoch_loss_count)

            # Debug info averaging
            for k in dbg_acc.keys():
                dbg_acc[k] /= max(1, valid_steps)

            # Logging
            if self.is_main and ((global_step % self.cfg.log_every) == 0 or global_step <= 5):
                self.tb.add_scalar("train/loss_total", train_loss_avg, global_step)
                
                # 더 자세한 로깅
                prog = (global_step % self.steps_per_epoch) / self.steps_per_epoch if self.steps_per_epoch > 0 else 0.0
                pct = int(prog * 100)
                line = (
                    f"Enhanced Epoch {epoch}: {pct:3d}% | step {global_step}/{self.cfg.max_steps} "
                    f"| loss={train_loss_avg:.5f} "
                    f"| t_range=[{dbg_acc['t_min']:.0f},{dbg_acc['t_max']:.0f}] "
                    f"| mse={dbg_acc['mse_raw']:.5f} "
                    f"| hole_ratio={dbg_acc['hole_ratio']:.3f} "
                    f"| target_var={dbg_acc['target_var']:.4f}"
                )
                pbar.set_postfix_str(f"loss={train_loss_avg:.5f}")
                self.logger.info(line)
            
            # 추가적인 디버깅을 위한 강제 프리뷰 저장 로직
            if self.is_main and ((global_step % self.cfg.image_every) == 0 or global_step <= 5):
                try:
                    batch_vis = next(data_iter)
                    self._save_enhanced_preview(batch_vis, global_step, max_rows=min(4, self.cfg.batch_size))
                    pbar.write(f"[img] saved preview at step {global_step}")
                    self.logger.info(f"[img] saved preview at step {global_step}")
                except Exception as e:
                    msg = f"[warn] preview save failed: {e}"
                    pbar.write(msg); 
                    self.logger.info(msg)


            # Synchronization
            if self.is_dist:
                dist.barrier()

            # Epoch 관리
            if (global_step % self.steps_per_epoch) == 0:
                epoch += 1
                if self.is_main and (epoch % self.cfg.save_epoch_ckpt) == 0:
                    path = self._save_ckpt(epoch, train_loss_avg, global_step)
                    self.logger.info(f"Epoch checkpoint saved: {path}")
                
                # 에포크 통계 리셋
                self._epoch_loss_sum = 0.0
                self._epoch_loss_count = 0
                
                if self.is_main:
                    pbar.set_description(f"Enhanced Epoch {epoch}")
                
                if self.is_dist and self.sampler is not None:
                    self.sampler.set_epoch(epoch)

        # 최종 체크포인트 저장
        final_loss = self._epoch_loss_sum / max(1, self._epoch_loss_count) if self._epoch_loss_count > 0 else 0.0
        if self.is_main:
            final_path = self._save_ckpt(epoch, final_loss, global_step)
            self.logger.info(f"Final checkpoint saved: {final_path}")
        
        pbar.close()
        if self.is_main:
            self.tb.close()
            self.logger.info("Enhanced training completed successfully")
        
        if self.is_dist:
            dist.barrier()


# ------------------------------------------------------------
# Enhanced Config & CLI
# ------------------------------------------------------------
ENHANCED_DEFAULTS = {
    "list_file": None,
    "sd3_model": "stabilityai/stable-diffusion-3.5-large",
    "size_h": 512, "size_w": 384,
    "mask_based": True, "invert_mask": True,

    # 향상된 학습 설정
    "lr": 3e-5,  # 더 안정적인 학습률
    "batch_size": 6, "grad_accum": 2, "max_steps": 128000,
    "seed": 1337, "num_workers": 4,
    "mixed_precision": "bf16",  # bf16 권장 (더 안정적)
    "use_scaler": False,  # bf16에서는 스케일러 불필요

    # 메모리 최적화
    "prefer_xformers": True,
    "disable_text": False,
    "save_root_dir": "logs", "save_name": "enhanced_catvton_sd35",
    "log_every": 25, "image_every": 250,  # 더 자주 로깅
    "save_epoch_ckpt": 10,  # 더 자주 저장
    "default_resume_ckpt": None,
    "resume_ckpt": None,

    # SD3.5 최적화된 학습 파라미터
    "weighting_scheme": "logit_normal",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "loss_sigma_weight": True,

    # 향상된 프리뷰 설정
    "preview_infer_steps": 20,  # 더 빠른 프리뷰
    "preview_seed": 1234,
    "preview_strength": 0.8,  # 더 강한 디노이징
    "preview_rows": 2,
    "preview_caption_h": 32,

    # 향상된 프롬프트
    "fixed_prompt": "high quality, photorealistic, detailed clothing texture, natural lighting, sharp focus",
    "disable_dynamic_shifting": False,

    # 런타임
    "hf_token": None,
}

def parse_enhanced_args():
    p = argparse.ArgumentParser(description="Enhanced SD3.5 CatVTON Training")
    p.add_argument("--config", type=str, default="configs/claude.yaml", help="YAML config path")
    p.add_argument("--list_file", type=str, default="DATA/VITON-HD/vitonhd_train_mask.csv", help="Training data list file")
    p.add_argument("--sd3_model", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="SD3.5 model")
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
    p.add_argument("--weighting_scheme", type=str, default=None, 
                   choices=["sigma_sqrt","logit_normal","mode","cosmap"])
    p.add_argument("--logit_mean", type=float, default=None)
    p.add_argument("--logit_std", type=float, default=None)
    p.add_argument("--mode_scale", type=float, default=None)
    p.add_argument("--fixed_prompt", type=str, default=None)
    p.add_argument("--disable_dynamic_shifting", action="store_true")
    p.add_argument("--enable_dynamic_shifting", action="store_true")
    return p.parse_args()

def load_enhanced_config(args: argparse.Namespace) -> DotDict:
    cfg = dict(ENHANCED_DEFAULTS)

    # YAML 설정 로드
    if args.config and os.path.isfile(args.config):
        if yaml is None:
            raise RuntimeError("PyYAML required. Install: pip install pyyaml")
        with open(args.config, "r") as f:
            y = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in y.items() if v is not None})

    # CLI 오버라이드
    for k in list(cfg.keys()):
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None and not isinstance(v, bool):
                cfg[k] = v

    # 불린 플래그 처리
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

    if getattr(args, "enable_dynamic_shifting", False):
        cfg["disable_dynamic_shifting"] = False
    elif getattr(args, "disable_dynamic_shifting", False):
        cfg["disable_dynamic_shifting"] = True

    if getattr(args, "fixed_prompt", None) is not None:
        cfg["fixed_prompt"] = args.fixed_prompt

    return DotDict(cfg)

def build_enhanced_run_dirs(cfg: DotDict, run_name: Optional[str] = None, create: bool = True) -> Dict[str, str]:
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_{cfg.save_name}"
    run_dir = os.path.join(cfg.save_root_dir, run_name)
    paths = {
        "run_dir": run_dir,
        "images": os.path.join(run_dir, "images"),
        "models": os.path.join(run_dir, "models"),
        "tb": os.path.join(run_dir, "tb"),
        "logs": os.path.join(run_dir, "logs"),  # 추가 로그 디렉토리
    }
    if create:
        for d in paths.values():
            os.makedirs(d, exist_ok=True)
    return paths

def main():
    args = parse_enhanced_args()
    cfg = load_enhanced_config(args)

    # 분산 학습 초기화
    dinfo = maybe_init_distributed()
    is_dist = dinfo["is_dist"]
    rank = dinfo["rank"]
    

    # 실행 디렉토리 생성
    run_name = None
    if rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{ts}_{cfg.save_name}"
    run_name = bcast_object(run_name, src=0)
    run_dirs = build_enhanced_run_dirs(cfg, run_name=run_name, create=(rank == 0))

    if rank == 0:
        print("=" * 80)
        print("Enhanced SD3.5 CatVTON Training Script v2.0")
        print("=" * 80)
        _try_copy_running_script(run_dirs["run_dir"])
        print(f"Run directory: {run_dirs['run_dir']}")
        print(f"Enhanced training configuration:")
        print(f"  - Model: {cfg.sd3_model}")
        print(f"  - Resolution: {cfg.size_h}x{cfg.size_w}")
        print(f"  - Batch size: {cfg.batch_size} (grad_accum: {cfg.grad_accum})")
        print(f"  - Learning rate: {cfg.lr}")
        print(f"  - Mixed precision: {cfg.mixed_precision}")
        print(f"  - Max steps: {cfg.max_steps}")
        print(f"  - Preview every: {cfg.image_every} steps")

    cfg_yaml_to_save = dict(cfg) if rank == 0 else None

    # 향상된 트레이너 생성 및 실행
    trainer = EnhancedCatVTON_SD3_Trainer(cfg, run_dirs, cfg_yaml_to_save=cfg_yaml_to_save)
    
    try:
        trainer.train()
        if rank == 0:
            print("\n" + "=" * 80)
            print("Enhanced training completed successfully!")
            print("=" * 80)
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        if is_dist:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
            