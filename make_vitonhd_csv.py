# make_vitonhd_csv.py
import os, csv, argparse
from PIL import Image

DEFAULT_MASK_DIR = "agnostic-mask"
V3_DIR = "agnostic-v3.2"
CANDIDATE_EXTS = [".jpg", ".png", ".jpeg"]  # jpg 우선 탐색

# -----------------------------
# Small helpers
# -----------------------------
def _stem_from_rel(rel: str) -> str:
    """ 'image/08996_00.jpg' -> '08996_00' """
    return os.path.splitext(os.path.basename(rel))[0]

def _resolve_with_exts(base_dir: str, stem: str, preferred_ext: str = None):
    """base_dir/stem.{ext} 중 존재하는 첫 파일 경로 반환"""
    exts = []
    if preferred_ext:
        exts.append(preferred_ext.lower())
    exts.extend([e for e in CANDIDATE_EXTS if e.lower() != (preferred_ext or "").lower()])
    for ext in exts:
        p = os.path.join(base_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def find_mask_by_stem(root_split_dir: str, stem: str, allowed_dirs, preferred_ext=".png"):
    """
    train/ 또는 test/ 아래에서 allowed_dirs 내에서 마스크 경로 탐색.
    보편적 형태: agnostic-mask/{stem}_mask.png
    예외적으로:   agnostic-mask/{stem}.png, agnostic-v3.2/{stem}.jpg 등
    """
    name_variants = [
        f"{stem}_mask",          # 가장 흔함
        stem,                    # 간혹 접미사 없는 경우
        f"{stem}_agnostic",
        f"{stem}_agnostic-mask",
    ]

    # agnostic-mask엔 보통 바로 파일이 있고, 드물게 agnostic-mask/image/에 있을 수도 있어 안전하게 둘 다 본다.
    candidate_subdirs = ["", "image"]

    # 1) 우선순위: allowed_dirs 순서
    for d in allowed_dirs:
        base_root = os.path.join(root_split_dir, d)
        for sub in candidate_subdirs:
            base_dir = os.path.join(base_root, sub) if sub else base_root
            for name in name_variants:
                p = _resolve_with_exts(base_dir, name, preferred_ext)
                if p:
                    return p
    return None

def is_binary_mask(path, tolerance_levels=4):
    """0/255 위주인지 대략 검사 (중간톤 < 1%)"""
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return False
    hist = im.histogram()
    lo = sum(hist[:tolerance_levels+1])
    hi = sum(hist[255-tolerance_levels:])
    mid = sum(hist[tolerance_levels+1:255-tolerance_levels])
    tot = lo + hi + mid
    if tot == 0:
        return False
    return (mid / tot) < 0.01

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build CSV for VITON-HD (paired-by-left).")
    ap.add_argument("--root", required=True, help="VITON-HD root (e.g., DATA/VITON-HD)")
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--mode", choices=["mask","maskfree"], default="mask",
                    help="'mask' -> 3 columns; 'maskfree' -> 2 columns")
    ap.add_argument("--mask_dirname", default=None,
                    help=f"force a mask folder name (default: {DEFAULT_MASK_DIR})")
    ap.add_argument("--mask_ext", default=".png", help="preferred mask extension")
    ap.add_argument("--strict", action="store_true", help="error out if any file missing")
    ap.add_argument("--fallback_to_v3", action="store_true",
                    help=f"also search '{V3_DIR}' for mask fallback")
    ap.add_argument("--check_mask", action="store_true",
                    help="warn if mask seems non-binary")
    args = ap.parse_args()

    pairs_txt = os.path.join(args.root, f"{args.split}_pairs.txt")
    split_dir = os.path.join(args.root, args.split)
    person_dir = os.path.join(split_dir, "image")
    cloth_dir  = os.path.join(split_dir, "cloth")

    assert os.path.isfile(pairs_txt), f"missing pairs: {pairs_txt}"
    assert os.path.isdir(person_dir), f"missing dir: {person_dir}"
    assert os.path.isdir(cloth_dir),  f"missing dir: {cloth_dir}"

    # 마스크 탐색 디렉터리 구성
    if args.mask_dirname:
        allowed_dirs = [args.mask_dirname]
    else:
        allowed_dirs = [DEFAULT_MASK_DIR]
        if args.fallback_to_v3:
            allowed_dirs.append(V3_DIR)

    n_all = n_ok = 0
    with open(pairs_txt) as fin, open(args.out, "w", newline="") as fout:
        w = csv.writer(fout)
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            n_all += 1

            # pairs.txt는 "왼쪽 파일명  오른쪽 파일명" 형태지만,
            # 여기서는 **왼쪽만 사용**하여 person/cloth/mask를 모두 같은 stem으로 매칭한다.
            try:
                left_rel = line.split()[0]
            except Exception:
                if args.strict:
                    raise
                print(f"[warn] bad line: {line}")
                continue

            stem = _stem_from_rel(left_rel)

            # person / cloth 경로 (왼쪽 stem으로 동일하게 구성)
            person_path = _resolve_with_exts(person_dir, stem, preferred_ext=".jpg")
            cloth_path  = _resolve_with_exts(cloth_dir,  stem, preferred_ext=".jpg")

            if person_path is None or cloth_path is None:
                if args.strict:
                    raise FileNotFoundError(f"missing person/cloth for stem={stem}")
                print(f"[warn] missing person/cloth for stem={stem}")
                continue

            if args.mode == "maskfree":
                w.writerow([person_path, cloth_path])
                n_ok += 1
                continue

            # mask 경로 (왼쪽 stem 기반)
            mask_path = find_mask_by_stem(split_dir, stem, allowed_dirs, preferred_ext=args.mask_ext)
            if mask_path is None:
                msg = f"mask not found for stem={stem} (dirs={allowed_dirs})"
                if args.strict:
                    raise FileNotFoundError(msg)
                print(f"[warn] {msg}")
                continue

            if args.check_mask and not is_binary_mask(mask_path):
                print(f"[warn] mask seems non-binary: {mask_path} (expect cloth=0, others=1)")

            w.writerow([person_path, cloth_path, mask_path])
            n_ok += 1

    print(f"[done] wrote {args.out} ({n_ok}/{n_all} rows) | mode={args.mode} | paired-by-left | masks from {allowed_dirs}")

if __name__ == "__main__":
    main()
