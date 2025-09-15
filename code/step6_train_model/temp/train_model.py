#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6: Train Segmentation Model
================================

- Expects Step 5 output:
  {step5_dir}/train/video_xx/{images/*.jpg|png, RGB_segmentation/*.png|jpg}
  {step5_dir}/validation/video_xx/{images/*.jpg|png, RGB_segmentation/*.png|jpg}

- Trains a ResNet34-UNet with selective Attention Gates on skip 1 & 2 (configurable).
- CUDA + AMP + DataParallel (when available).
- Loss = 0.5 * Soft IoU (Jaccard) + 0.5 * NSD-like Surface loss (τ = 3 px by default).
- Background [0,0,0] is EXCLUDED (not a class).
- Resumes automatically from the latest checkpoint if present.
- Saves per-video & overall mIoU and NSD metrics each epoch to CSV.

Author: ProjectMach
"""

import os
import re
import json
import math
import glob
import time
import yaml
import random
import torch.nn.functional as F
import shutil
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# -----------------------
# Utils: filesystem
# -----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_videos(split_dir: str) -> List[str]:
    return sorted([d for d in glob.glob(os.path.join(split_dir, "video_*")) if os.path.isdir(d)])

def list_images(img_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([f for f in glob.glob(os.path.join(img_dir, "*")) if f.lower().endswith(exts)])

def stem_only(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

# Robust match of image to mask via stem
def make_stem_index(files: List[str]) -> Dict[str, str]:
    idx = {}
    for f in files:
        idx[stem_only(f).lower()] = f
    return idx

# -----------------------
# Color map handling
# -----------------------

def parse_rgb_key(k: str) -> Tuple[int, int, int]:
    parts = [int(x) for x in k.split('-')]
    assert len(parts) == 3
    return tuple(parts)

def discover_colors(mask_path: str, bg_rgb: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
    arr = np.array(Image.open(mask_path).convert("RGB"))
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    colors = [tuple(map(int, c)) for c in colors]
    colors = [c for c in colors if c != tuple(bg_rgb)]
    return colors

# -----------------------
# Geometry helpers (NSD)
# -----------------------

def _distance_map_numpy(binary_mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform to the boundary.
    Returns distance-to-boundary for each pixel inside+outside using a simple trick:
    distance to edge ~ min(dist to foreground boundary, dist to background boundary)
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion
    # Boundary pixels (xor of mask and eroded mask)
    eroded = binary_erosion(binary_mask, iterations=1, border_value=0)
    boundary = binary_mask ^ eroded
    # Distance to boundary from inside:
    dist_in = distance_transform_edt(~boundary)
    # We want distance from both sides; approximate by:
    # distance to boundary = distance to nearest boundary pixel
    return dist_in

def _make_surface_band_torch(gt: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Create a soft 'band' mask around GT boundaries using distance transform.
    This relies on numpy distance (CPU) once per batch element & class (OK for τ small,
    and we cache per epoch via no_grad metric path; for loss, we use a soft approx).
    gt: [B, C, H, W] binary {0,1}
    returns band: [B, C, H, W] in [0,1] (soft mask via sigmoid around tau)
    """
    B, C, H, W = gt.shape
    bands = []
    gt_np = gt.detach().cpu().numpy().astype(bool)
    for b in range(B):
        chan = []
        for c in range(C):
            dm = _distance_map_numpy(gt_np[b, c])
            # Soft threshold around tau: sigmoid(-(d - tau)) -> high near <= tau
            # scale factor to tighten transition around tau
            s = 4.0
            soft = 1.0 / (1.0 + np.exp(s * (dm - tau)))
            chan.append(torch.from_numpy(soft).to(gt.device, dtype=torch.float32))
        bands.append(torch.stack(chan, dim=0))
    return torch.stack(bands, dim=0)

# -----------------------
# Losses
# -----------------------

class SoftIoULoss(nn.Module):
    """ Soft Jaccard over classes (exclude empty GT channels to stabilize) """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # probs: [B,C,H,W], target: [B,C,H,W] one-hot
        dims = (0,2,3)
        inter = (probs * target).sum(dims)
        union = (probs + target - probs*target).sum(dims) + self.eps
        iou = inter / union
        # mask classes with no GT to avoid noise
        valid = target.sum(dims) > 0
        iou = torch.where(valid, iou, torch.zeros_like(iou))
        # mean over valid classes and batch
        denom = valid.sum().clamp_min(1)
        return 1.0 - (iou.sum() / denom)

class NSDLikeSurfaceLoss(nn.Module):
    """
    NSD-inspired surface loss: encourage high probability within a τ-px band
    around GT boundaries, using a soft band via distance transform + sigmoid.
    """
    def __init__(self, tau_px: float = 3.0):
        super().__init__()
        self.tau = float(tau_px)

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # probs: [B,C,H,W], target: [B,C,H,W] one-hot
        with torch.no_grad():
            band = _make_surface_band_torch(target, self.tau)  # [B,C,H,W] in [0,1]
        # High prob in band for the correct class -> maximize (probs * band * target)
        # Turn into loss by 1 - average score over banded GT
        num = (probs * band * target).sum()
        den = (band * target).sum().clamp_min(1.0)
        score = num / den
        return 1.0 - score

# -----------------------
# Metrics (exact mIoU & NSD on binarized preds)
# -----------------------

def compute_iou_per_class(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # pred, gt: [H,W] with {0,1} for a class
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return np.nan
    return inter / union

def boundary_pixels(binary_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary_mask, iterations=1, border_value=0)
    return np.logical_xor(binary_mask, eroded)

def nsd_for_class(pred: np.ndarray, gt: np.ndarray, tau: float) -> float:
    """
    Exact NSD per class (Seidlitz style): fraction of boundary pixels within τ of the other.
    """
    from scipy.ndimage import distance_transform_edt

    bp_pred = boundary_pixels(pred)
    bp_gt = boundary_pixels(gt)

    if bp_pred.sum() == 0 and bp_gt.sum() == 0:
        return np.nan

    # Distance maps to opposite boundaries
    # For each boundary pixel in A, distance to nearest in B is distance_transform on ~B, sampled at A
    dist_to_gt = distance_transform_edt(~bp_gt)
    dist_to_pred = distance_transform_edt(~bp_pred)

    if bp_pred.sum() > 0:
        d_pred = dist_to_gt[bp_pred]
        tp_pred = (d_pred <= tau).sum()
        denom_pred = d_pred.size
    else:
        tp_pred, denom_pred = 0, 0

    if bp_gt.sum() > 0:
        d_gt = dist_to_pred[bp_gt]
        tp_gt = (d_gt <= tau).sum()
        denom_gt = d_gt.size
    else:
        tp_gt, denom_gt = 0, 0

    num = tp_pred + tp_gt
    den = denom_pred + denom_gt
    if den == 0:
        return np.nan
    return num / den

# -----------------------
# Dataset
# -----------------------

class VideoSegDataset(Dataset):
    """
    Loads (image, mask) pairs across all videos in a split.
    RGB masks are mapped to class indices excluding background color.
    """

    def __init__(self, split_dir: str, class_colors: List[Tuple[int,int,int]],
                 bg_rgb: Tuple[int,int,int], out_size: Tuple[int,int]):
        super().__init__()
        self.out_w, self.out_h = out_size
        self.bg_rgb = tuple(bg_rgb)
        self.class_colors = [tuple(c) for c in class_colors]  # list of RGB triplets
        # map RGB->channel index (0..C-1), excluding background
        self.rgb2chan = {tuple(c): i for i, c in enumerate(self.class_colors)}

        self.samples = []  # (img_path, mask_path, video_id, frame_stem)
        videos = list_videos(split_dir)
        for vdir in videos:
            img_dir = os.path.join(vdir, "images")
            msk_dir = os.path.join(vdir, "RGB_segmentation")
            if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
                continue
            imgs = list_images(img_dir)
            msks = list_images(msk_dir)
            if not imgs or not msks:
                continue
            idx_m = make_stem_index(msks)
            for ip in imgs:
                st = stem_only(ip).lower()
                if st in idx_m:
                    self.samples.append((ip, idx_m[st], os.path.basename(vdir), st))

        # Simple transforms (keep deterministic/scalable)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.samples)

    def _resize(self, pil_img: Image.Image, size: Tuple[int,int], is_mask=False) -> Image.Image:
        if is_mask:
            return pil_img.resize(size, Image.NEAREST)
        return pil_img.resize(size, Image.BILINEAR)

    def _mask_to_onehot(self, mask_rgb: np.ndarray) -> np.ndarray:
        # mask_rgb: [H,W,3], returns [C,H,W] with background excluded
        H, W, _ = mask_rgb.shape
        C = len(self.class_colors)
        onehot = np.zeros((C, H, W), dtype=np.float32)
        # Fast vectorized match per class
        for ci, rgb in enumerate(self.class_colors):
            match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8)[None,None,:], axis=2)
            onehot[ci] = match.astype(np.float32)
        return onehot

    def __getitem__(self, idx: int):
        ip, mp, vid, stem = self.samples[idx]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("RGB")

        # Resize to training size
        img = self._resize(img, (self.out_w, self.out_h), is_mask=False)
        msk = self._resize(msk, (self.out_w, self.out_h), is_mask=True)

        img_t = self.to_tensor(img)  # [3,H,W]
        mnp = np.array(msk, dtype=np.uint8)
        onehot = self._mask_to_onehot(mnp)   # [C,H,W] excluding background
        y = torch.from_numpy(onehot)         # float32

        return img_t, y, vid, stem

# -----------------------
# Model: ResNet34-UNet with selective Attention Gates
# -----------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AttentionGate(nn.Module):
    """ Simple additive attention gate for a skip connection """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder feature (gating), x: encoder skip feature
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class UNetResNet34(nn.Module):
    def __init__(self, n_classes: int, base_ch=32, pretrained=True, attention_skips=(1,2)):
        super().__init__()
        self.attention_skips = set(attention_skips)

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # Encoder stages
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # 64, /2 after maxpool
        self.pool0 = resnet.maxpool
        self.enc1 = resnet.layer1  # 64
        self.enc2 = resnet.layer2  # 128
        self.enc3 = resnet.layer3  # 256
        self.enc4 = resnet.layer4  # 512

        # Channel dims from encoder
        chs = [64, 64, 128, 256, 512]

        # Decoder
        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = nn.Sequential(ConvBNReLU(chs[3]+chs[3], chs[3]), ConvBNReLU(chs[3], chs[3]))

        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = nn.Sequential(ConvBNReLU(chs[2]+chs[2], chs[2]), ConvBNReLU(chs[2], chs[2]))

        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = nn.Sequential(ConvBNReLU(chs[1]+chs[1], chs[1]), ConvBNReLU(chs[1], chs[1]))

        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = nn.Sequential(ConvBNReLU(chs[0]+chs[0], chs[0]), ConvBNReLU(chs[0], chs[0]))

        self.up0 = nn.ConvTranspose2d(chs[0], base_ch, 2, stride=2)
        self.dec0 = nn.Sequential(ConvBNReLU(base_ch, base_ch), ConvBNReLU(base_ch, base_ch))

        # Attention gates for selected skips (indices align with decoder depth: 1..3)
        self.ag3 = AttentionGate(F_g=chs[3], F_l=chs[3], F_int=chs[3]//2)  # between enc3 and dec4
        self.ag2 = AttentionGate(F_g=chs[2], F_l=chs[2], F_int=chs[2]//2)  # enc2 -> dec3
        self.ag1 = AttentionGate(F_g=chs[1], F_l=chs[1], F_int=chs[1]//2)  # enc1 -> dec2

        self.head = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape  # save original input size

        # ---------- Encoder ----------
        x0 = self.enc0(x)  # [B,64,H/2,W/2]
        p0 = self.pool0(x0)  # [B,64,H/4,W/4]
        x1 = self.enc1(p0)  # [B,64,H/4,W/4]
        x2 = self.enc2(x1)  # [B,128,H/8,W/8]
        x3 = self.enc3(x2)  # [B,256,H/16,W/16]
        x4 = self.enc4(x3)  # [B,512,H/32,W/32]

        # ---------- Decoder ----------
        d4 = self.up4(x4)
        d4 = F.interpolate(d4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        s3 = self.ag3(d4, x3) if 3 in self.attention_skips else x3
        d4 = torch.cat([d4, s3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        s2 = self.ag2(d3, x2) if 2 in self.attention_skips else x2
        d3 = torch.cat([d3, s2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        s1 = self.ag1(d2, x1) if 1 in self.attention_skips else x1
        d2 = torch.cat([d2, s1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x0.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = F.interpolate(d0, size=(H, W), mode="bilinear", align_corners=False)
        d0 = self.dec0(d0)

        logits = self.head(d0)  # [B,C,H,W]
        return logits


# -----------------------
# Training / Eval
# -----------------------

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def build_class_colors(cfg: dict, split_dir: str, bg_rgb: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
    cmap = cfg["data"].get("color_map", {})
    if cmap:
        colors = [parse_rgb_key(k) for k in cmap.keys()]
        return colors
    # auto-discover from first available mask in split
    for vdir in list_videos(split_dir):
        mdir = os.path.join(vdir, "RGB_segmentation")
        mlist = list_images(mdir)
        if mlist:
            disc = discover_colors(mlist[0], bg_rgb)
            if not disc:
                continue
            return disc
    raise RuntimeError("Could not discover class colors; please fill data.color_map in config.")

def make_loaders(cfg: dict, class_colors: List[Tuple[int,int,int]], bg_rgb: Tuple[int,int,int]):
    root = cfg["data"]["step5_dir"]
    tdir = os.path.join(root, cfg["data"]["train_dirname"])
    vdir = os.path.join(root, cfg["data"]["val_dirname"])
    size = tuple(cfg["data"]["train_size"])  # (W,H)

    train_ds = VideoSegDataset(tdir, class_colors, bg_rgb, size)
    val_ds   = VideoSegDataset(vdir, class_colors, bg_rgb, size)

    ld_cfg = cfg["loader"]
    train_ld = DataLoader(
        train_ds,
        batch_size=ld_cfg["batch_size"],
        shuffle=ld_cfg.get("shuffle", True),
        num_workers=ld_cfg.get("num_workers", 4),
        pin_memory=ld_cfg.get("pin_memory", True),
        drop_last=False,
        persistent_workers=(ld_cfg.get("num_workers", 4) > 0)
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=max(1, ld_cfg["batch_size"]),
        shuffle=False,
        num_workers=ld_cfg.get("num_workers", 4),
        pin_memory=ld_cfg.get("pin_memory", True),
        drop_last=False,
        persistent_workers=(ld_cfg.get("num_workers", 4) > 0)
    )
    return train_ld, val_ld

def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)

def onehot_from_probs(probs: torch.Tensor, thresh: float=0.5) -> torch.Tensor:
    # argmax -> hard one-hot
    arg = torch.argmax(probs, dim=1)  # [B,H,W]
    oh = F.one_hot(arg, num_classes=probs.shape[1]).permute(0,3,1,2).float()
    return oh

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def find_latest_checkpoint(ckpt_dir: str) -> str:
    files = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if not files:
        return ""
    def epnum(p):
        m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    files.sort(key=epnum, reverse=True)
    return files[0]

def evaluate(val_loader, model, device, class_names, tau_px, out_csv_path):
    model.eval()
    C = len(class_names)
    per_video = defaultdict(lambda: {"iou_sum": 0.0, "nsd_sum": 0.0, "frames": 0})

    with torch.no_grad():
        for imgs, ys, vids, stems in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)  # [B,C,H,W]

            logits = model(imgs)
            probs = probs_from_logits(logits)
            oh_pred = onehot_from_probs(probs).cpu().numpy()  # [B,C,H,W]
            oh_true = ys.cpu().numpy()

            B = oh_true.shape[0]
            for b in range(B):
                vid = vids[b]
                iou_list = []
                nsd_list = []
                for c in range(C):
                    pred = oh_pred[b, c].astype(bool)
                    gt   = oh_true[b, c].astype(bool)
                    iou_c = compute_iou_per_class(pred, gt)
                    nsd_c = nsd_for_class(pred, gt, tau_px)
                    if not math.isnan(iou_c):
                        iou_list.append(iou_c)
                    if not math.isnan(nsd_c):
                        nsd_list.append(nsd_c)

                miou = float(np.mean(iou_list)) if iou_list else 0.0
                mnsd = float(np.mean(nsd_list)) if nsd_list else 0.0

                per_video[vid]["iou_sum"] += miou
                per_video[vid]["nsd_sum"] += mnsd
                per_video[vid]["frames"]  += 1

    # Write per-video and overall
    ensure_dir(os.path.dirname(out_csv_path))
    import csv
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frames", "mIoU_video", "mNSD_video"])
        total_iou = 0.0
        total_nsd = 0.0
        V = 0
        for vid, rec in sorted(per_video.items()):
            frames = max(1, rec["frames"])
            miou_v = rec["iou_sum"] / frames
            mnsd_v = rec["nsd_sum"] / frames
            w.writerow([vid, frames, f"{miou_v:.6f}", f"{mnsd_v:.6f}"])
            total_iou += miou_v
            total_nsd += mnsd_v
            V += 1
        if V > 0:
            w.writerow(["__OVERALL__", "", f"{(total_iou/V):.6f}", f"{(total_nsd/V):.6f}"])

    # Return overall scores for scheduler/model selection
    if V == 0:
        return 0.0, 0.0
    return total_iou / V, total_nsd / V

def main():
    # 1) Load config
    cfg_path = os.path.join("D:/ProjectMach", "config", "step6_config.yaml")
    cfg = load_config(cfg_path)

    seed_all(cfg["train"]["seed"])

    # 2) Resolve paths
    out_root = cfg["output"]["step6_dir"]
    ckpt_dir = os.path.join(out_root, cfg["output"]["checkpoints_dirname"])
    met_dir  = os.path.join(out_root, cfg["output"]["metrics_dirname"])
    ensure_dir(ckpt_dir); ensure_dir(met_dir)

    # 3) Class colors (exclude background)
    step5_train_dir = os.path.join(cfg["data"]["step5_dir"], cfg["data"]["train_dirname"])
    bg_rgb = tuple(cfg["data"]["background_rgb"])
    class_colors = build_class_colors(cfg, step5_train_dir, bg_rgb)
    class_names = []
    if cfg["data"].get("color_map", {}):
        # Preserve order given in config
        for k, v in cfg["data"]["color_map"].items():
            if parse_rgb_key(k) in class_colors:
                class_names.append(v)
        if len(class_names) != len(class_colors):
            # Fallback names
            class_names = [f"class_{i}" for i in range(len(class_colors))]
    else:
        class_names = [f"class_{i}" for i in range(len(class_colors))]

    print(f"[INFO] Classes ({len(class_colors)}): {class_names}")
    print(f"[INFO] Training size: {cfg['data']['train_size']} (W,H). Background excluded: {bg_rgb}")

    # 4) Dataloaders
    train_loader, val_loader = make_loaders(cfg, class_colors, bg_rgb)

    # 5) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34(
        n_classes=len(class_colors),
        base_ch=cfg["model"]["base_channels"],
        pretrained=cfg["model"]["pretrained"],
        attention_skips=tuple(cfg["model"]["attention_skips"])
    )
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 6) Optim & losses
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    loss_iou = SoftIoULoss()
    loss_nsd = NSDLikeSurfaceLoss(tau_px=cfg["train"]["nsd_tau_px"])

    start_epoch = 1
    best_score = -1.0

    # 7) Auto-resume if checkpoint exists
    latest = find_latest_checkpoint(ckpt_dir)
    if latest and os.path.isfile(latest):
        print(f"[RESUME] Loading checkpoint: {latest}")
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", -1.0)
    else:
        print("[RESUME] No previous checkpoint found. Starting fresh.")

    # 8) Train loop
    epochs = cfg["train"]["epochs"]
    w_iou = float(cfg["train"]["loss_weights"]["iou"])
    w_nsd = float(cfg["train"]["loss_weights"]["nsd"])
    grad_clip = float(cfg["train"]["grad_clip"])
    log_every = int(cfg["train"]["log_every_steps"])
    tau_px = float(cfg["train"]["nsd_tau_px"])

    log_path = os.path.join(out_root, "train_log.jsonl")
    with open(log_path, "a") as logf:

        for epoch in range(start_epoch, epochs+1):
            model.train()
            t0 = time.time()
            running = {"loss": 0.0, "iou": 0.0, "nsd": 0.0}
            steps = 0

            for i, (imgs, ys, vids, stems) in enumerate(train_loader, start=1):
                imgs = imgs.to(device, non_blocking=True)
                ys   = ys.to(device, non_blocking=True)  # [B,C,H,W] float 0/1

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                    logits = model(imgs)
                    probs = probs_from_logits(logits)
                    liou = loss_iou(probs, ys)
                    lnsd = loss_nsd(probs, ys)
                    loss = w_iou * liou + w_nsd * lnsd

                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()

                running["loss"] += float(loss.detach().cpu().item())
                running["iou"]  += float(liou.detach().cpu().item())
                running["nsd"]  += float(lnsd.detach().cpu().item())
                steps += 1

                if i % log_every == 0:
                    print(f"[E{epoch:03d}][{i:05d}] loss={running['loss']/steps:.4f} "
                          f"iouL={running['iou']/steps:.4f} nsdL={running['nsd']/steps:.4f}")

            # 9) Eval each epoch
            metrics_csv = os.path.join(met_dir, f"epoch_{epoch:03d}.csv")
            miou_overall, mnsd_overall = evaluate(
                val_loader, model, device, class_names, tau_px, metrics_csv
            )

            elapsed = time.time() - t0
            print(f"[E{epoch:03d}] val mIoU={miou_overall:.4f}  mNSD={mnsd_overall:.4f}  time={elapsed/60:.1f}m")

            # 10) Save checkpoints
            state = {
                "epoch": epoch,
                "model": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "classes": class_names,
            }
            save_checkpoint(state, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
            save_checkpoint(state, os.path.join(ckpt_dir, cfg["output"]["last_name"]))

            # Best based on (mIoU + mNSD)/2
            score = 0.5 * (miou_overall + mnsd_overall)
            if score > best_score:
                best_score = score
                save_checkpoint(state, os.path.join(ckpt_dir, cfg["output"]["best_name"]))
                print(f"[BEST] New best score={best_score:.4f} saved.")

            # 11) Log JSONL
            log_rec = {
                "epoch": epoch,
                "train_loss": running["loss"]/max(1,steps),
                "train_iou_loss": running["iou"]/max(1,steps),
                "train_nsd_loss": running["nsd"]/max(1,steps),
                "val_mIoU": miou_overall,
                "val_mNSD": mnsd_overall,
                "elapsed_sec": elapsed,
                "classes": class_names
            }
            logf.write(json.dumps(log_rec) + "\n")
            logf.flush()

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
