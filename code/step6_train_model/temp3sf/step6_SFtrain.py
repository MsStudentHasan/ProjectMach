#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6 – SegFormer training using Step 5 splits
===============================================
- Uses Step 5 outputs for train and validation (no on-the-fly split).
- Expects masks in RGB color format under 'RGB_segmentation'.
- Trains at 960x540; validates at original resolution with TTA.
- Saves 'last.pth' (full), 'best.pth' (full-by-score), and periodic 'epoch_XX.pth' (model-only).
- Supports resume from either a full checkpoint or a model-only .pth.

Place:
  config: D:/ProjectMach/config/step6_SFconfig.yaml
  code  : D:/ProjectMach/code/step6_train_model/step6_SFtrain.py
Outputs:
  D:/ProjectMach/output/output_step_06/segformer/<timestamp>/
"""

import os, re, math, json, time, random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from transformers import SegformerForSemanticSegmentation
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

# -----------------------
# CONFIG LOADER
# -----------------------
class Cfg:
    pass

def load_cfg(yaml_path: str) -> Cfg:
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)

    c = Cfg()
    # Project
    c.SAVE_DIR     = y["project"]["save_dir"]
    c.SEED         = int(y["project"].get("seed", 123))
    # Data
    c.TRAIN_ROOT   = y["data"]["train_root"]
    c.VAL_ROOT     = y["data"]["val_root"]
    c.RESIZE_TO    = tuple(y["data"]["resize_to"])      # (W,H)
    c.NUM_WORKERS  = int(y["data"]["num_workers"])
    c.BATCH_SIZE   = int(y["data"]["batch_size"])
    c.NORM_MEAN    = y["data"]["norm_mean"]
    c.NORM_STD     = y["data"]["norm_std"]
    # Model
    c.SEGFORMER_MODEL = y["model"]["segformer_name"]
    c.N_CLASSES       = int(y["model"]["n_classes"])
    # Optim/Train
    c.EPOCHS       = int(y["optim"]["epochs"])
    c.LR           = float(y["optim"]["lr"])
    c.WEIGHT_DECAY = float(y["optim"]["weight_decay"])
    c.AMP          = bool(y["optim"]["amp"])
    c.SAVE_EVERY   = int(y["optim"]["save_every"])
    c.RESUME_FROM  = y["optim"].get("resume_from") or ""
    # Eval
    c.TTA_FLIP     = bool(y["eval"]["tta_flip"])
    c.TTA_ROT_DEG  = int(y["eval"]["tta_rot_deg"])
    c.NSD_TAU_PIX  = int(y["eval"]["nsd_tau_pix"])
    # Classes (RGB map)
    c.CLASSES = {int(k): tuple(v) for k, v in y["classes"].items()}
    return c

# -----------------------
# UTILITIES
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_videos(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def _find_mask_for_image(mask_dir: Path, img_name: str) -> Path:
    """Match mask by basename; prefer .png."""
    stem = Path(img_name).stem
    candidates = [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}.jpeg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path()

def find_pairs(video_dir: Path) -> List[Tuple[Path, Path, str]]:
    img_dir = video_dir / "images"
    seg_dir = video_dir / "RGB_segmentation"  # <— Step 3/5 output
    assert img_dir.is_dir() and seg_dir.is_dir(), f"Missing images/RGB_segmentation in {video_dir}"
    items = []
    for img_path in sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg"), *img_dir.glob("*.jpeg")]):
        mask_path = _find_mask_for_image(seg_dir, img_path.name)
        if mask_path.exists():
            items.append((img_path, mask_path, video_dir.name))
    return items

def rgb_mask_to_index(mask_rgb: np.ndarray, rgb_map: Dict[int, Tuple[int,int,int]]) -> np.ndarray:
    """Convert RGB-coded mask (H,W,3) to class indices (H,W)."""
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls, color in rgb_map.items():
        matches = (mask_rgb == np.array(color, dtype=np.uint8)).all(axis=-1)
        out[matches] = cls
    return out

def to_tensor_img(img: np.ndarray, mean, std) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    mean = torch.tensor(mean).view(3,1,1); std = torch.tensor(std).view(3,1,1)
    return (t - mean) / std

def apply_same_geo_transforms(img_pil: Image.Image, mask_np: np.ndarray, resize_to=(960,540), train=True):
    """Resize + random HFlip + small random rotation (train). Mask is class-index (H,W)."""
    new_w, new_h = resize_to
    img_pil_r = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    mask_pil  = Image.fromarray(mask_np, mode='L').resize((new_w, new_h), resample=Image.NEAREST)

    if train and random.random() < 0.5:
        img_pil_r = TF.hflip(img_pil_r)
        mask_pil  = TF.hflip(mask_pil)

    if train:
        deg = random.uniform(-10, 10)
        img_pil_r = TF.rotate(img_pil_r, deg, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask_pil  = TF.rotate(mask_pil,  deg, interpolation=TF.InterpolationMode.NEAREST,  fill=0)

    img_np = np.array(img_pil_r, dtype=np.uint8)
    mask_np2 = np.array(mask_pil, dtype=np.uint8)
    return img_np, mask_np2

def up_or_downsample_logits(logits: torch.Tensor, size_hw):
    return F.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)

# -----------------------
# DATASET / DATALOADER
# -----------------------
class SurgDataset(Dataset):
    """
    Returns:
      img_t:     (3,h,w) float tensor (standardized)
      mask_t:    (h,w) long indices (resized for training)
      vid:       video name (str)
      img_path:  str path to original image
      msk_path:  str path to original mask (for full-res metrics)
      orig_hw:   (H,W) tensor for the original image size
    """
    def __init__(self, items: List[Tuple[Path,Path,str]], rgb_map: Dict[int,Tuple[int,int,int]], mean, std, resize_to, train=True):
        self.items = items
        self.rgb_map = rgb_map
        self.train = train
        self.mean = mean
        self.std = std
        self.resize_to = resize_to

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, msk_path, vid = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        mask_rgb = np.array(Image.open(msk_path).convert('RGB'), dtype=np.uint8)
        mask_idx = rgb_mask_to_index(mask_rgb, self.rgb_map)

        orig_size = (img.height, img.width)  # (H,W)
        img_np, mask_np = apply_same_geo_transforms(img, mask_idx, resize_to=self.resize_to, train=self.train)

        img_t  = to_tensor_img(img_np, self.mean, self.std)  # CHW
        mask_t = torch.from_numpy(mask_np.astype(np.int64))  # HW (long)

        return img_t, mask_t, vid, str(img_path), str(msk_path), torch.tensor(orig_size, dtype=torch.long)

# -----------------------
# MODEL (SegFormer with custom head)
# -----------------------
class SegFormerModel(nn.Module):
    def __init__(self, model_name="nvidia/mit-b2", n_classes=10):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=n_classes,
            ignore_mismatched_sizes=True
        )
    def forward(self, x):
        out = self.segformer(pixel_values=x)
        logits = out.logits
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits

# -----------------------
# LOSSES (unchanged)
# -----------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=self.n_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        intersection = torch.sum(probs * target_1h, dims)
        cardinality = torch.sum(probs + target_1h, dims)
        dice = (2.*intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class CEDice(nn.Module):
    def __init__(self, n_classes, ce_weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = SoftDiceLoss(n_classes)
    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)

# -----------------------
# METRICS (IoU, NSD, Score)
# -----------------------
def per_class_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    ious = np.zeros(n_classes, dtype=np.float64)
    for k in range(n_classes):
        pred_k = (pred == k); gt_k = (gt == k)
        inter = np.logical_and(pred_k, gt_k).sum()
        union = np.logical_or(pred_k, gt_k).sum()
        ious[k] = 1.0 if union == 0 else inter / union
    return ious

def boundary_map(bin_mask: np.ndarray) -> np.ndarray:
    if bin_mask.sum() == 0:
        return np.zeros_like(bin_mask, dtype=bool)
    er = binary_erosion(bin_mask, iterations=1, border_value=0)
    return np.logical_xor(bin_mask, er)

def nsd_per_class(pred: np.ndarray, gt: np.ndarray, k: int, tau: int) -> float:
    pb = boundary_map(pred == k)
    gb = boundary_map(gt == k)
    n_pb = int(pb.sum()); n_gb = int(gb.sum())
    if n_pb == 0 and n_gb == 0: return 1.0
    if n_pb == 0 and n_gb > 0:  return 0.0
    if n_gb == 0 and n_pb > 0:  return 0.0
    dist_to_gb = distance_transform_edt(~gb)
    dist_to_pb = distance_transform_edt(~pb)
    n_valid_pred = int((dist_to_gb[pb] <= tau).sum())
    n_valid_gt   = int((dist_to_pb[gb] <= tau).sum())
    return (n_valid_pred + n_valid_gt) / (n_pb + n_gb)

def mIoU_frame(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> float:
    return float(per_class_iou(pred, gt, n_classes).mean())

def mNSD_frame(pred: np.ndarray, gt: np.ndarray, n_classes: int, tau: int) -> float:
    return float(np.mean([nsd_per_class(pred, gt, k, tau) for k in range(n_classes)]))

# -----------------------
# TTA INFERENCE (val)
# -----------------------
@torch.no_grad()
def predict_with_tta(model, img_t: torch.Tensor, orig_hw, device, rot_deg: int, do_flip: bool):
    model.eval()
    aug_inps, inv_ops = [], []
    aug_inps.append(img_t.clone()); inv_ops.append(lambda y: y)

    for d in (+rot_deg, -rot_deg):
        aug_inps.append(TF.rotate(img_t, d, interpolation=TF.InterpolationMode.BILINEAR))
        inv_ops.append(lambda y, dd=d: TF.rotate(y, -dd, interpolation=TF.InterpolationMode.BILINEAR))

    if do_flip:
        x = TF.hflip(img_t); aug_inps.append(x); inv_ops.append(lambda y: TF.hflip(y))

    logits_sum = None
    for x, inv in zip(aug_inps, inv_ops):
        x1 = x.unsqueeze(0).to(device)
        logits = model(x1)  # (1,C,h,w)
        logits_np = logits.squeeze(0).cpu()
        logits_inv = inv(logits_np).unsqueeze(0)
        logits_inv = up_or_downsample_logits(logits_inv, orig_hw)
        logits_sum = logits_inv if logits_sum is None else logits_sum + logits_inv

    return logits_sum / len(aug_inps)  # (1,C,H0,W0)

# -----------------------
# TRAIN / EVAL
# -----------------------
def train_one_epoch(model, loader, optim, scaler, device, criterion, use_amp: bool):
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        img, msk = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = criterion(logits, msk)
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        else:
            logits = model(img); loss = criterion(logits, msk)
            loss.backward(); optim.step()
        running += loss.item() * img.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, n_classes: int, tau: int, rot_deg: int, do_flip: bool):
    model.eval()
    video_frame_mIoU, video_frame_mNSD = defaultdict(list), defaultdict(list)

    for batch in tqdm(loader, desc="valid", leave=False):
        img, _, vids, img_paths, msk_paths, orig_hw = batch
        for b in range(img.size(0)):
            x = img[b]; vid = vids[b]
            H0, W0 = int(orig_hw[b][0].item()), int(orig_hw[b][1].item())

            gt_rgb = np.array(Image.open(msk_paths[b]).convert('RGB'), dtype=np.uint8)
            gt_full = rgb_mask_to_index(gt_rgb, C.CLASSES)

            logits_full = predict_with_tta(model, x, (H0, W0), device, rot_deg, do_flip)
            pred_full = logits_full.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            miou = mIoU_frame(pred_full, gt_full, n_classes)
            mnsd = mNSD_frame(pred_full, gt_full, n_classes, tau)
            video_frame_mIoU[vid].append(miou); video_frame_mNSD[vid].append(mnsd)

    if len(video_frame_mIoU) == 0:
        return dict(mIoU=0.0, mNSD=0.0, score=0.0)

    per_video_mIoU = {v: float(np.mean(vals)) for v, vals in video_frame_mIoU.items()}
    per_video_mNSD = {v: float(np.mean(vals)) for v, vals in video_frame_mNSD.items()}
    mIoU = float(np.mean(list(per_video_mIoU.values())))
    mNSD = float(np.mean(list(per_video_mNSD.values())))
    score = float(np.sqrt(max(mIoU, 0.0) * max(mNSD, 0.0)))
    return dict(mIoU=mIoU, mNSD=mNSD, score=score)

# -----------------------
# RESUME HELPERS (kept the same behavior)
# -----------------------
def _epoch_from_filename(p: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pth$", p.name)
    return int(m.group(1)) if m else 0

def try_resume(resume_path: Path, model, optimizer, scaler, device):
    ckpt = torch.load(resume_path, map_location=device)
    run_dir = resume_path.parent
    start_epoch = 0
    best_score = -1.0

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0))
        # optimizer/scaler optional
        try:
            optimizer.load_state_dict(ckpt["opt"]); print("-> Optimizer state restored.")
        except Exception:
            print("-> Optimizer state not found; starting fresh optimizer.")
        try:
            scaler.load_state_dict(ckpt["scaler"]); print("-> AMP scaler state restored.")
        except Exception:
            print("-> AMP scaler state not found; starting fresh scaler.")
        prev_metrics = ckpt.get("metrics")
        if isinstance(prev_metrics, dict) and "score" in prev_metrics:
            best_score = float(prev_metrics["score"])
        print(f"Resumed full checkpoint from '{resume_path}' (epoch {start_epoch}).")
    else:
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise RuntimeError(f"Unsupported checkpoint type in {resume_path}")
        start_epoch = _epoch_from_filename(resume_path)
        print(f"Loaded model weights from '{resume_path}'. Starting from epoch {start_epoch}+1 with fresh optimizer/scaler.")

    return start_epoch, run_dir, best_score

# -----------------------
# MAIN
# -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to step6_SFconfig.yaml")
    args = parser.parse_args()

    global C
    C = load_cfg(args.config)
    set_seed(C.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Build data from Step 5 split folders
    train_vids = list_videos(Path(C.TRAIN_ROOT))
    val_vids   = list_videos(Path(C.VAL_ROOT))
    assert len(train_vids) > 0 and len(val_vids) > 0, "Train/Val folders from Step 5 must have video_xx subfolders."

    train_items, val_items = [], []
    for v in train_vids: train_items += find_pairs(v)
    for v in val_vids:   val_items   += find_pairs(v)

    print(f"[INFO] Train frames: {len(train_items)} | Val frames: {len(val_items)}")
    ds_train = SurgDataset(train_items, C.CLASSES, C.NORM_MEAN, C.NORM_STD, C.RESIZE_TO, train=True)
    ds_val   = SurgDataset(val_items,   C.CLASSES, C.NORM_MEAN, C.NORM_STD, C.RESIZE_TO, train=False)

    dl_train = DataLoader(ds_train, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val, batch_size=1, shuffle=False,
                          num_workers=max(1, C.NUM_WORKERS//2), pin_memory=True, drop_last=False)

    # Model / loss / opt
    print(f"[INFO] Loading SegFormer: {C.SEGFORMER_MODEL}")
    model = SegFormerModel(model_name=C.SEGFORMER_MODEL, n_classes=C.N_CLASSES).to(device)
    criterion = CEDice(n_classes=C.N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=C.AMP)

    # Resume / run dir
    resume_path = Path(C.RESUME_FROM) if C.RESUME_FROM else None
    if resume_path and resume_path.exists():
        start_epoch, run_dir, best_score = try_resume(resume_path, model, optimizer, scaler, device)
    else:
        start_epoch, best_score = 0, -1.0
        run_dir = Path(C.SAVE_DIR) / time.strftime("%Y%m%d_%H%M%S")
        ensure_dir(run_dir)
        # persist effective config
        with open(run_dir / "config_effective.json", "w") as f:
            json.dump({
                "SAVE_DIR": C.SAVE_DIR,
                "TRAIN_ROOT": C.TRAIN_ROOT,
                "VAL_ROOT": C.VAL_ROOT,
                "RESIZE_TO": C.RESIZE_TO,
                "BATCH_SIZE": C.BATCH_SIZE,
                "EPOCHS": C.EPOCHS,
                "LR": C.LR,
                "WEIGHT_DECAY": C.WEIGHT_DECAY,
                "NUM_WORKERS": C.NUM_WORKERS,
                "AMP": C.AMP,
                "SEED": C.SEED,
                "SEGFORMER_MODEL": C.SEGFORMER_MODEL,
                "N_CLASSES": C.N_CLASSES,
                "TTA_ROT_DEG": C.TTA_ROT_DEG,
                "TTA_FLIP": C.TTA_FLIP,
                "NSD_TAU_PIX": C.NSD_TAU_PIX,
            }, f, indent=2)

    print(f"[INFO] Run dir: {run_dir}")
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_mIoU,val_mNSD,val_score,ckpt\n")

    if start_epoch >= C.EPOCHS:
        print(f"[INFO] start_epoch ({start_epoch}) >= EPOCHS ({C.EPOCHS}). Nothing to train.")
        return

    print("[INFO] Starting training...")
    for epoch in range(start_epoch + 1, C.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{C.EPOCHS} (resumed from {start_epoch})")
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device, criterion, C.AMP)
        print(f"  train loss: {train_loss:.4f}")

        metrics = evaluate(model, dl_val, device, C.N_CLASSES, C.NSD_TAU_PIX, C.TTA_ROT_DEG, C.TTA_FLIP)
        print(f"  val mIoU: {metrics['mIoU']:.4f} | val mNSD: {metrics['mNSD']:.4f} | score: {metrics['score']:.4f}")

        ckpt_flag = ""
        # Save "last" (full)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": metrics
        }, run_dir / "last.pth")

        # Save best-by-score (full)
        if metrics['score'] > best_score:
            best_score = metrics['score']
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "metrics": metrics
            }, run_dir / "best.pth")
            ckpt_flag = "best.pth"

        # Periodic model-only snapshot
        if epoch % C.SAVE_EVERY == 0:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch}.pth")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{metrics['mIoU']:.6f},{metrics['mNSD']:.6f},{metrics['score']:.6f},{ckpt_flag}\n")

    print(f"\n[INFO] Training done. Best score: {best_score:.4f}")
    print(f"[INFO] Run dir: {run_dir}")

if __name__ == "__main__":
    main()
