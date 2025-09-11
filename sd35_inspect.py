#!/usr/bin/env python3
# sd35_inspect.py
# SD 3.5를 CPU(기본) 또는 GPU로 로드하고 모듈 트리를 상세 덤프.
# - 기본: CPU + Transformer만 스캔 + QKV 시뮬
# - --include-pipe: 파이프라인 내부의 모든 nn.Module을 각각 순회(접두사 붙여 출력)

import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn

try:
    from diffusers import StableDiffusion3Pipeline
except Exception as e:
    raise SystemExit(
        "diffusers가 필요합니다. 설치: pip install -U 'diffusers>=0.34.0' 'transformers>=4.41.0' 'safetensors>=0.4.3'"
    ) from e

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------- utils ----------------
def fmt_count(n): return f"{n/1e6:.2f}M"

def dtype_from_str(s: str, device: str):
    s = (s or "auto").lower()
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    if s == "fp32": return torch.float32
    # auto
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def safe_getattr(obj, name, default=None):
    try: return getattr(obj, name)
    except Exception: return default

def is_cross_attention_module(mod, name: str):
    is_cross = safe_getattr(mod, "is_cross_attention", None)
    if isinstance(is_cross, bool):
        return is_cross
    proc = safe_getattr(mod, "processor", None)
    proc_is_cross = safe_getattr(proc, "is_cross_attention", None)
    if isinstance(proc_is_cross, bool):
        return proc_is_cross
    lname = name.lower()
    if any(tok in lname for tok in ("attn2", "cross", "encoder")):
        return True
    return False

def has_qkv(mod: nn.Module):
    return all(hasattr(mod, a) for a in ("to_q", "to_k", "to_v", "to_out"))

def list_qkv_shapes(mod: nn.Module):
    info = {}
    for sub in ("to_q", "to_k", "to_v", "to_out"):
        m = safe_getattr(mod, sub, None)
        if m is None:
            info[sub] = None
            continue
        w = safe_getattr(m, "weight", None)
        b = safe_getattr(m, "bias", None)
        info[sub] = {
            "w": tuple(w.shape) if isinstance(w, torch.Tensor) else None,
            "b": tuple(b.shape) if isinstance(b, torch.Tensor) else None,
        }
    return info

def is_adaln_like(name: str):
    n = name.lower()
    return any(k in n for k in ("adaln", "ada_ln", "ada_norm", "ada", "modulation", "to_scale", "to_shift", "to_gate"))

def count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def print_write(s, fh):
    print(s, flush=True)
    if fh is not None:
        fh.write(s + "\n")
        fh.flush()

class TeeStd:
    def __init__(self, path):
        self._fh = open(path, "w", encoding="utf-8")
        self._out = sys.__stdout__
        self._err = sys.__stderr__
    def write(self, s):
        self._out.write(s); self._out.flush()
        self._fh.write(s);  self._fh.flush()
    def flush(self):
        self._out.flush(); self._fh.flush()
    def close(self):
        try: self._fh.close()
        except Exception: pass

def simulate_qkv_freeze(transformer: nn.Module, include_adaln=False):
    saved = {n: p.requires_grad for n, p in transformer.named_parameters()}
    try:
        for p in transformer.parameters():
            p.requires_grad = False
        kept_names = []
        for mname, mod in transformer.named_modules():
            if is_cross_attention_module(mod, mname):
                continue
            if has_qkv(mod):
                for subn in ("to_q", "to_k", "to_v", "to_out"):
                    subm = safe_getattr(mod, subn, None)
                    if subm is not None:
                        for pn, p in subm.named_parameters(recurse=True):
                            p.requires_grad = True
                            kept_names.append(f"{mname}.{subn}.{pn}")
                continue
            if include_adaln and is_adaln_like(mname):
                for pn, p in mod.named_parameters(recurse=True):
                    p.requires_grad = True
                    kept_names.append(f"{mname}.{pn}")
        t_all, t_tr = count_params(transformer)
        return kept_names, t_all, t_tr
    finally:
        for n, p in transformer.named_parameters():
            p.requires_grad = saved[n]

def iter_pipeline_modules_with_prefix(pipe) -> list:
    """파이프라인 객체 내의 모든 nn.Module을 (prefix, module)로 나열."""
    pairs = []
    # 1) 잘 알려진 컴포넌트 우선 추가(있을 때만)
    for key in (
        "transformer", "vae",
        "text_encoder", "text_encoder_2", "text_encoder_3",
        "image_encoder", "unet", "prior", "projector",
    ):
        mod = safe_getattr(pipe, key, None)
        if isinstance(mod, nn.Module):
            pairs.append((key, mod))
    # 2) 그 외: 속성 탐색
    for k, v in vars(pipe).items():
        if isinstance(v, nn.Module) and all(k != p[0] for p in pairs):
            pairs.append((k, v))
        # ModuleList/ModuleDict도 nn.Module이므로 위에서 잡힘
    # 중복 제거(동일 모듈 객체 id 기준)
    seen = set()
    unique = []
    for prefix, mod in pairs:
        if id(mod) in seen: continue
        seen.add(id(mod))
        unique.append((prefix, mod))
    return unique

# ---------------- main ----------------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda")  # 기본을 cpu 로 변경
    ap.add_argument("--dtype",  type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--filter", type=str, default=None, help="이 문자열이 포함된 모듈만 출력")
    ap.add_argument("--include-pipe", action="store_true", help="파이프라인 전체를 스캔")
    ap.add_argument("--no-simulate", dest="simulate", action="store_false", help="Q/K/V/Out 시뮬레이션 끄기")
    ap.add_argument("--include_adaln", action="store_true", help="시뮬레이션 시 AdaLN/Modulation 포함")
    ap.add_argument("--no-xformers", dest="xformers", action="store_false", help="xFormers 비활성화")
    ap.add_argument("--out", type=str, default=None, help="레이어 덤프 파일(.txt)")
    ap.add_argument("--log", type=str, default=None, help="콘솔 로그 파일(.log)")
    ap.add_argument("--no-log", action="store_true", help="콘솔 로그 저장 비활성화")
    ap.set_defaults(simulate=True, xformers=True)

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 가용 GPU가 없어 CPU로 강제 전환합니다.", flush=True)
        device = "cpu"
    dt = dtype_from_str(args.dtype, device)

    out_dir = "sd35_dumps"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, f"sd35_layers_{ts}.txt")
    log_path = None if args.no_log else (args.log or os.path.join(out_dir, f"sd35_check_{ts}.log"))

    tee = None
    try:
        if log_path is not None:
            tee = TeeStd(log_path)
            sys.stdout = tee
            sys.stderr = tee

        print(f"[run] model={args.model} device={device} dtype={dt}")
        print(f"[out] dump={out_path}")
        print(f"[scope] include_pipe={args.include_pipe} | filter={args.filter!r}")
        if log_path: print(f"[log] tee={log_path}")

        print("[load] downloading/loading pipeline...", flush=True)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model,
            torch_dtype=dt,
            use_safetensors=True,
            local_files_only=False,
        )
        pipe = pipe.to(device)

        if args.xformers and device == "cuda":
            try:
                pipe.transformer.enable_xformers_memory_efficient_attention()
                print("[xformers] enabled memory-efficient attention")
            except Exception as e:
                print(f"[xformers] enable failed: {e}")

        transformer = pipe.transformer
        p_all, p_tr = count_params(transformer)
        print(f"[transformer] total params={fmt_count(p_all)} trainable(now)={fmt_count(p_tr)}")
        if device == "cuda":
            torch.cuda.synchronize()
            print(f"[cuda] memory_allocated={torch.cuda.memory_allocated()/1e9:.2f} GB | reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

        fh = open(out_path, "w", encoding="utf-8")
        fh.write(f"# SD3.5 dump for '{args.model}'  ({ts})\n")

        # 스캔 대상 구성
        scan_targets = []
        if args.include_pipe:
            # 파이프라인 내부 모든 nn.Module 수집
            for prefix, mod in iter_pipeline_modules_with_prefix(pipe):
                scan_targets.append((prefix, mod))
        else:
            scan_targets.append(("transformer", transformer))

        stats = defaultdict(int)
        print_write("\n=== MODULE TREE (filtered) ===", fh)

        # 각 대상 모듈별로 접두사 붙여 순회
        for root_prefix, root_mod in scan_targets:
            for name, mod in root_mod.named_modules():
                full_name = f"{root_prefix}" if name == "" else f"{root_prefix}.{name}"
                if args.filter and (args.filter.lower() not in full_name.lower()):
                    continue

                cls = mod.__class__.__name__
                cross = is_cross_attention_module(mod, full_name)
                qkv = has_qkv(mod)
                adaln = is_adaln_like(full_name)

                line = f"{full_name} :: {cls}"
                marks = []
                if qkv: marks.append("QKV")
                if cross: marks.append("CROSS")
                if adaln: marks.append("ADALN?")
                if marks:
                    line += " [" + ",".join(marks) + "]"
                print_write(line, fh)

                if qkv:
                    shp = list_qkv_shapes(mod)
                    for k in ("to_q","to_k","to_v","to_out"):
                        info = shp.get(k)
                        print_write(f"  - {k:6s}: W={info['w']}  B={info['b']}", fh)

                stats["modules"] += 1
                if qkv:   stats["qkv_modules"] += 1
                if cross: stats["cross_modules"] += 1
                if adaln: stats["adaln_like"] += 1

        print_write("\n=== SUMMARY ===", fh)
        print_write(f"total scanned modules: {stats['modules']}", fh)
        print_write(f"QKV modules:          {stats['qkv_modules']}", fh)
        print_write(f"CROSS-labeled modules:{stats['cross_modules']}", fh)
        print_write(f"AdaLN-like (name):    {stats['adaln_like']}  (이건 이름 기반 휴리스틱)", fh)

        if args.simulate:
            kept, tall, ttr = simulate_qkv_freeze(transformer, include_adaln=args.include_adaln)
            print_write("\n=== SIMULATE: freeze_all_but_self_attn_qkv ===", fh)
            print_write(f"include_adaln={args.include_adaln}", fh)
            print_write(f"transformer params total={fmt_count(tall)} | would-be-trainable={fmt_count(ttr)}", fh)
            show = kept[:40]
            print_write(f"would keep {len(kept)} parameter tensors (show up to 40):", fh)
            for k in show:
                print_write("  - " + k, fh)

        fh.close()
        print(f"\n[done] wrote detailed dump to: {out_path}")
        if log_path:
            print(f"[done] console log saved to: {log_path}")

    finally:
        if tee is not None:
            tee.flush()
            tee.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
