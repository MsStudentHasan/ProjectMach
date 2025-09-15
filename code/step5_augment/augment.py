#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5: Data Augmentation (RGB only, CUDA/Parallel, robust ID matching)
=======================================================================

What this script does
---------------------
- Reads ONLY from Step 3: .../output_step_03/{train,test}/video_xx/{images,RGB_segmentation}
- Uses Step 4 split_assignments.csv to decide whether each Step-3 *train* video
  should be written under Step 5's output/train or output/validation.
- Test is copied over unchanged (no augmentation).
- Aug policy:
    • Compute per-video class appearance counts (#frames with >0 px per class).
    • Compute per-video per-class avg pixels.
    • Identify least 3 classes by appearances -> least2 + third.
    • For each frame:
        - If any of least2 class pixels > that class's avg → apply ALL 5 augs
        - Else if third class pixels > its avg → apply DOUBLE (2 random augs)
        - Else → SINGLE (1 random aug)
- Robust frame ID normalization: matches CSV image_id to filenames regardless of
  zero-padding or extension.

Performance
-----------
- Optional CUDA acceleration via torch/torchvision when available (backend="auto"/"torch").
- Multiprocessing / thread pools for copying and augmenting frames.
- tqdm per-video progress with augmentation mode displayed.

Windows-friendly
----------------
- Uses absolute paths from YAML. Safe joins and existence checks.

Folder I/O (no surprises)
-------------------------
Input (Step 3):
  {root}/output/output_step_03/
    train/video_xx/images/*.png|jpg
    train/video_xx/RGB_segmentation/*.png
    test/video_xx/images/*.png|jpg
    test/video_xx/RGB_segmentation/*.png

Output (Step 5):
  {root}/output/output_step_05/
    train/video_xx/{images,RGB_segmentation}
    validation/video_xx/{images,RGB_segmentation}
    test/...  (copied unchanged)

"""

import os
import io
import yaml
import math
import shutil
import random
import pathlib
import functools
import concurrent.futures as futures

import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Optional Torch backend
try:
    import torch
    import torchvision.transforms.functional as Fv
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


# ----------------------------- Helpers -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_cfg():
    # project root is 2 levels above this file: /code/step5_augment/augment.py
    here = pathlib.Path(__file__).resolve()
    project_root = here.parents[2]
    cfg_path = project_root / "config" / "step5_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_id(raw_id):
    """
    Normalize CSV id to match filenames:
    - strip extension if any
    - strip left zeros → keep "0" if all zeros
    """
    s = str(raw_id)
    s = os.path.splitext(s)[0]
    stripped = s.lstrip("0")
    return stripped if stripped != "" else "0"

def build_basename_index(img_names):
    """
    Build maps for quick matching of normalized IDs to basenames.
    Returns:
      - basenames (list, no ext)
      - index_by_nostrip (dict exact basename → basename)
      - index_by_stripped (dict stripped-leftzeros → list of basenames matching when left zeros removed)
    """
    basenames = [os.path.splitext(fn)[0] for fn in img_names]
    idx_exact = {b: b for b in basenames}
    idx_strip = {}
    for b in basenames:
        k = b.lstrip("0")
        if k == "": k = "0"
        idx_strip.setdefault(k, []).append(b)
    return basenames, idx_exact, idx_strip

def match_basename(norm_id, idx_exact, idx_strip):
    """
    Try exact match first, then left-zero stripped bucket, then suffix/prefix heuristics.
    """
    # exact
    if norm_id in idx_exact:
        return norm_id

    # left-zero stripped bucket
    if norm_id in idx_strip and len(idx_strip[norm_id]) == 1:
        return idx_strip[norm_id][0]
    if norm_id in idx_strip and len(idx_strip[norm_id]) > 1:
        # choose the one with closest length (stability heuristic)
        candidates = idx_strip[norm_id]
        return min(candidates, key=lambda b: abs(len(b) - len(norm_id)))

    # final heuristic: find any basename that endswith(norm_id) or norm_id.endswith(b)
    # (supports dataset quirks like 'frame_000123' vs '123' or vice versa)
    for b in idx_exact.keys():
        if b.endswith(norm_id) or norm_id.endswith(b):
            return b

    return None

def pil_rotate(img, mask, max_angle):
    angle = random.uniform(-max_angle, max_angle)
    tag = "rotatePlus" if angle >= 0 else "rotateMinus"
    return img.rotate(angle, Image.BILINEAR), mask.rotate(angle, Image.NEAREST), tag

def pil_brightness(img, mask, delta):
    d = random.uniform(-delta, delta)
    factor = 1.0 + d
    tag = "brightnessPlus" if d >= 0 else "brightnessMinus"
    return ImageEnhance.Brightness(img).enhance(factor), mask, tag

def pil_zoom(img, mask, zf):
    d = random.uniform(-zf, zf)
    w, h = img.size
    cw, ch = int(w * (1 - abs(d))), int(h * (1 - abs(d)))
    cw, ch = max(1, cw), max(1, ch)
    left = random.randint(0, max(0, w - cw))
    top = random.randint(0, max(0, h - ch))
    box = (left, top, left + cw, top + ch)
    tag = "zoomIn" if d > 0 else "zoomOut"
    return img.crop(box).resize((w, h), Image.BILINEAR), \
           mask.crop(box).resize((w, h), Image.NEAREST), tag

def pil_blur(img, mask, radius):
    return img.filter(ImageFilter.GaussianBlur(radius)), mask, "blur"

def pil_flip(img, mask):
    return ImageOps.mirror(img), ImageOps.mirror(mask), "flip"


# ---- Torch variants (work on CPU or CUDA) ----
def _to_tensors(img_pil, mask_pil, device):
    img = torch.frombuffer(img_pil.tobytes(), dtype=torch.uint8)
    img = img.view(img_pil.size[1], img_pil.size[0], 3).permute(2, 0, 1).float() / 255.0
    if mask_pil.mode != "RGB":
        mask_pil = mask_pil.convert("RGB")
    mask = torch.frombuffer(mask_pil.tobytes(), dtype=torch.uint8)
    mask = mask.view(mask_pil.size[1], mask_pil.size[0], 3).permute(2, 0, 1).float() / 255.0
    return img.to(device), mask.to(device)

def _to_pil(img_t, mask_t):
    img = (img_t.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    mask = (mask_t.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img, mode="RGB"), Image.fromarray(mask, mode="RGB")

def tv_rotate(img_pil, mask_pil, max_angle, device):
    angle = random.uniform(-max_angle, max_angle)
    tag = "rotatePlus" if angle >= 0 else "rotateMinus"
    img, mask = _to_tensors(img_pil, mask_pil, device)
    img2 = Fv.rotate(img, angle, interpolation=torchvision_interpolation_bilinear())
    mask2 = Fv.rotate(mask, angle, interpolation=torchvision_interpolation_nearest())
    return _to_pil(img2, mask2) + (tag,)

def tv_brightness(img_pil, mask_pil, delta, device):
    d = random.uniform(-delta, delta)
    factor = 1.0 + d
    tag = "brightnessPlus" if d >= 0 else "brightnessMinus"
    img, mask = _to_tensors(img_pil, mask_pil, device)
    img2 = Fv.adjust_brightness(img, factor)
    return _to_pil(img2, mask) + (tag,)

def tv_zoom(img_pil, mask_pil, zf, device):
    d = random.uniform(-zf, zf)
    w, h = img_pil.size
    cw, ch = int(w * (1 - abs(d))), int(h * (1 - abs(d)))
    cw, ch = max(1, cw), max(1, ch)
    left = random.randint(0, max(0, w - cw))
    top = random.randint(0, max(0, h - ch))
    right, bottom = left + cw, top + ch
    tag = "zoomIn" if d > 0 else "zoomOut"

    img, mask = _to_tensors(img_pil, mask_pil, device)
    # crop via slicing then resize back
    img_crop = img[:, top:bottom, left:right]
    mask_crop = mask[:, top:bottom, left:right]
    img2 = Fv.resize(img_crop, [h, w], antialias=True)
    mask2 = Fv.resize(mask_crop, [h, w], antialias=False)
    return _to_pil(img2, mask2) + (tag,)

def tv_blur(img_pil, mask_pil, radius, device):
    # torchvision GaussianBlur expects kernel size (odd int) and sigma
    k = max(1, int(radius) * 2 + 1)
    img, mask = _to_tensors(img_pil, mask_pil, device)
    img2 = Fv.gaussian_blur(img, kernel_size=[k, k])
    return _to_pil(img2, mask) + ("blur",)

def tv_flip(img_pil, mask_pil, device):
    img, mask = _to_tensors(img_pil, mask_pil, device)
    img2 = torch.flip(img, dims=[2])
    mask2 = torch.flip(mask, dims=[2])
    return _to_pil(img2, mask2) + ("flip",)

def torchvision_interpolation_bilinear():
    # handle version differences
    import torchvision
    if hasattr(torchvision.transforms, "InterpolationMode"):
        return torchvision.transforms.InterpolationMode.BILINEAR
    return 2  # PIL.Image.BILINEAR

def torchvision_interpolation_nearest():
    import torchvision
    if hasattr(torchvision.transforms, "InterpolationMode"):
        return torchvision.transforms.InterpolationMode.NEAREST
    return 0  # PIL.Image.NEAREST


# ----------------------------- Aug registry -----------------------------
def make_aug_fns(backend, device, aug_cfg):
    """
    Returns:
      AUG_ALL: list[(key, fn)]  (fn(img_pil, mask_pil) → (img_pil, mask_pil, tag))
    """
    if backend == "torch":
        AUG_ALL = [
            ("rotate",    lambda im, m: tv_rotate(im, m, aug_cfg["rotation"], device)),
            ("brightness",lambda im, m: tv_brightness(im, m, aug_cfg["brightness"], device)),
            ("zoom",      lambda im, m: tv_zoom(im, m, aug_cfg["zoom"], device)),
            ("blur",      lambda im, m: tv_blur(im, m, aug_cfg["blur"], device)),
            ("flip",      lambda im, m: tv_flip(im, m, device)),
        ]
    else:
        AUG_ALL = [
            ("rotate",    lambda im, m: pil_rotate(im, m, aug_cfg["rotation"])),
            ("brightness",lambda im, m: pil_brightness(im, m, aug_cfg["brightness"])),
            ("zoom",      lambda im, m: pil_zoom(im, m, aug_cfg["zoom"])),
            ("blur",      lambda im, m: pil_blur(im, m, aug_cfg["blur"])),
            ("flip",      lambda im, m: pil_flip(im, m)),
        ]
    return AUG_ALL


# ----------------------------- Per-frame worker -----------------------------
def _save_png(pil_img: Image.Image, out_path: str):
    # fast & deterministic save
    pil_img.save(out_path, format="PNG", compress_level=6)

def process_frame(task):
    """
    task:
      {
        'img_path', 'mask_path', 'out_img_dir', 'out_mask_dir',
        'img_id', 'mode', 'AUG_ALL', 'AUG_RANDOM', 'aug_cfg'
      }
    Returns: (img_id, mode, summary_string)
    """
    img = Image.open(task["img_path"]).convert("RGB")
    if os.path.isfile(task["mask_path"]):
        mask = Image.open(task["mask_path"]).convert("RGB")
    else:
        mask = Image.new("RGB", img.size, (0, 0, 0))

    mode = task["mode"]
    AUG_ALL = task["AUG_ALL"]
    AUG_RANDOM = task["AUG_RANDOM"]
    img_id = task["img_id"]
    out_img = task["out_img_dir"]
    out_mask = task["out_mask_dir"]

    if mode == "ALL5":
        tags = []
        for key, fn in AUG_ALL:
            img_aug, mask_aug, tag = fn(img, mask)
            _save_png(img_aug, os.path.join(out_img, f"{img_id}_{tag}.png"))
            _save_png(mask_aug, os.path.join(out_mask, f"{img_id}_{tag}.png"))
            tags.append(tag)
        return img_id, mode, ",".join(tags)

    elif mode == "DOUBLE2":
        keys = random.sample(AUG_RANDOM, 2)
        tags = []
        for key, fn in keys:
            img_aug, mask_aug, tag = fn(img, mask)
            _save_png(img_aug, os.path.join(out_img, f"{img_id}_{tag}.png"))
            _save_png(mask_aug, os.path.join(out_mask, f"{img_id}_{tag}.png"))
            tags.append(tag)
        return img_id, mode, ",".join(tags)

    else:  # SINGLE1
        key, fn = random.choice(AUG_RANDOM)
        img_aug, mask_aug, tag = fn(img, mask)
        _save_png(img_aug, os.path.join(out_img, f"{img_id}_{tag}.png"))
        _save_png(mask_aug, os.path.join(out_mask, f"{img_id}_{tag}.png"))
        return img_id, mode, tag


# ----------------------------- Main -----------------------------
def main():
    cfg = load_cfg()
    random.seed(int(cfg.get("performance", {}).get("seed", 1337)))

    root_dir = cfg["root_dir"]
    in_step3_dir = os.path.join(root_dir, cfg["input_dir_step3"])
    class_csv = os.path.join(root_dir, cfg["input_class_distribution"])
    split_csv = os.path.join(root_dir, cfg["input_split_assignments"])
    out_dir = os.path.join(root_dir, cfg["output_dir"])

    aug_cfg = cfg["augmentation"]
    perf = cfg.get("performance", {})
    backend_pref = perf.get("backend", "auto").lower()

    # Decide backend
    use_torch = False
    device = "cpu"
    if backend_pref == "torch":
        use_torch = _HAVE_TORCH and torch.cuda.is_available()
        device = "cuda" if use_torch else "cpu"
    elif backend_pref == "auto":
        use_torch = _HAVE_TORCH and torch.cuda.is_available()
        device = "cuda" if use_torch else "cpu"
    else:
        use_torch = False
        device = "cpu"

    backend = "torch" if use_torch else "pil"
    if use_torch:
        torch.manual_seed(int(perf.get("seed", 1337)))

    AUG_ALL = make_aug_fns(backend, device, aug_cfg)
    AUG_RANDOM = list(AUG_ALL)

    # Load CSVs
    class_dist = pd.read_csv(class_csv)
    split_assign = pd.read_csv(split_csv)
    class_cols = [c for c in class_dist.columns if c.startswith("class_")]

    # Ensure base output dirs
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "train"))
    ensure_dir(os.path.join(out_dir, "validation"))

    # Process videos per split_assign (train/validation only)
    for _, r in split_assign.iterrows():
        video_id = str(r["video_id"])
        split = str(r["split"]).lower()

        if split not in ("train", "validation", "test"):
            continue
        if split == "test":
            # We'll handle test as a whole later (copy unchanged)
            continue

        vrows = class_dist[class_dist.video_id.astype(str) == video_id]
        if vrows.empty:
            print(f"[WARN] No class stats found for video {video_id}. Skipping.")
            continue

        # Compute appearance counts and per-class averages (over frames of *this* video)
        appearance_counts = (vrows[class_cols] > 0).sum()
        avg_per_class = vrows[class_cols].mean()

        print(f"\n=== Video {video_id} → output split: {split} | frames: {len(vrows)} ===")
        print("Appearance counts (frames with >0 pixels):")
        for c in class_cols:
            print(f"  {c}: {int(appearance_counts[c])} frames")
        print("Average pixel counts per frame:")
        for c in class_cols:
            print(f"  {c}: {avg_per_class[c]:.2f}")

        # Least classes by appearance
        least3 = appearance_counts.nsmallest(3).index.tolist()
        if len(least3) < 3:
            # safety: if fewer than 3 classes present, pad using the smallest seen
            missing = 3 - len(least3)
            least3 += [least3[-1]] * missing
        least2, third = least3[:2], least3[2]
        print(f"--> Least 2: {least2}, Third: {third}")

        # Source folder (Step 3 only has train + test)
        src_split = "train"  # because validation doesn't exist in step 3
        video_src = os.path.join(in_step3_dir, src_split, video_id)
        img_dir = os.path.join(video_src, "images")
        mask_dir = os.path.join(video_src, "RGB_segmentation")

        if not os.path.isdir(img_dir):
            print(f"[WARN] images dir missing for {video_id}: {img_dir}  (skipping video)")
            continue

        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        img_files.sort()
        basenames, idx_exact, idx_strip = build_basename_index(img_files)

        # Flag frames per policy
        all_aug_frames, double_aug_frames = set(), set()
        for _, rowv in vrows.iterrows():
            norm_id = normalize_id(rowv["image_id"])
            matched = match_basename(norm_id, idx_exact, idx_strip)
            if not matched:
                continue

            # Decisions use per-class avg for least2/third
            if any(rowv[c] > avg_per_class[c] for c in least2):
                all_aug_frames.add(matched)
            elif rowv[third] > avg_per_class[third]:
                double_aug_frames.add(matched)

        single_count = len(img_files) - len(all_aug_frames) - len(double_aug_frames)
        print(f"Flagged in {video_id}: ALL5={len(all_aug_frames)}, DOUBLE2={len(double_aug_frames)}, SINGLE≈{max(0, single_count)}")

        # Prepare output dirs
        out_video = os.path.join(out_dir, split, video_id)
        out_img = os.path.join(out_video, "images")
        out_mask = os.path.join(out_video, "RGB_segmentation")
        ensure_dir(out_img); ensure_dir(out_mask)

        # Copy originals (parallel)
        def _copy_one(fn):
            src_i = os.path.join(img_dir, fn)
            shutil.copy2(src_i, os.path.join(out_img, fn))
            mname = os.path.splitext(fn)[0] + ".png"
            src_m = os.path.join(mask_dir, mname)
            if os.path.isfile(src_m):
                shutil.copy2(src_m, os.path.join(out_mask, mname))

        copy_workers = int(perf.get("copy_workers", 16))
        with futures.ThreadPoolExecutor(max_workers=copy_workers) as ex:
            list(tqdm(ex.map(_copy_one, img_files), total=len(img_files), desc=f"Copy originals [{video_id}]"))

        # Build per-frame tasks
        tasks = []
        for fn in img_files:
            img_id = os.path.splitext(fn)[0]
            mode = "SINGLE1"
            if img_id in all_aug_frames:
                mode = "ALL5"
            elif img_id in double_aug_frames:
                mode = "DOUBLE2"

            tasks.append({
                "img_path": os.path.join(img_dir, fn),
                "mask_path": os.path.join(mask_dir, img_id + ".png"),
                "out_img_dir": out_img,
                "out_mask_dir": out_mask,
                "img_id": img_id,
                "mode": mode,
                "AUG_ALL": AUG_ALL,
                "AUG_RANDOM": AUG_RANDOM,
                "aug_cfg": aug_cfg,
            })

        # Parallel augment
        worker_count = int(perf.get("workers", 8))
        desc = f"Augment [{video_id}] ({backend}{'@CUDA' if backend=='torch' and device=='cuda' else ''})"
        with futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
            pbar = tqdm(total=len(tasks), desc=desc, unit="img")
            for img_id, mode, tags in ex.map(process_frame, tasks):
                pbar.set_postfix({"mode": mode, "tags": tags})
                pbar.update(1)
            pbar.close()

    # Copy test unchanged (delete if exists to stay clean)
    test_src = os.path.join(in_step3_dir, "test")
    test_dst = os.path.join(out_dir, "test")
    if os.path.isdir(test_src):
        if os.path.exists(test_dst):
            shutil.rmtree(test_dst)
        shutil.copytree(test_src, test_dst)
        print("\n[Test] copied unchanged to Step 5 output.")

    print("\nStep 5 augmentation completed successfully.")


if __name__ == "__main__":
    main()
