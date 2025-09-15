#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7: SegFormer Evaluation
============================

- Loads trained checkpoint from Step 6.
- Runs inference on test dataset (Step 5 output).
- Applies TTA (±20° rotation, horizontal flip).
- Computes IoU, NSD, and overall score.
- Saves:
    - Predicted masks (index + color)
    - Visualization (Original | GT | Pred | Overlay)
    - Per-frame & per-class metrics
    - Overall summary CSV
"""

import os, time, yaml, random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt
from transformers import SegformerForSemanticSegmentation


# ------------------------------------------------
# UTILS
# ------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_videos(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def find_mask_dir(video_dir: Path) -> Path:
    for name in ["segmentation", "RGB_segmentation", "masks", "labels"]:
        d = video_dir / name
        if d.is_dir():
            return d
    return None


def find_pairs(video_dir: Path) -> List[Tuple[Path, Path]]:
    img_dir = video_dir / "images"
    seg_dir = find_mask_dir(video_dir)
    if not img_dir.is_dir() or seg_dir is None:
        print(f"[WARN] Skipping {video_dir.name} (no images or masks)")
        return []
    mask_by_stem = {p.stem: p for p in seg_dir.iterdir() if p.suffix.lower() in [".png", ".jpg"]}
    items = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in [".png", ".jpg"]:
            continue
        m = mask_by_stem.get(img_path.stem)
        if m:
            items.append((img_path, m))
    print(f"[INFO] {video_dir.name}: {len(items)} frames with masks")
    return items


def rgb_to_index(mask_rgb: np.ndarray, cmap: Dict[int, Tuple[int,int,int]]) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls, color in cmap.items():
        matches = (mask_rgb == np.array(color, dtype=np.uint8)).all(axis=-1)
        out[matches] = cls
    return out


def index_to_rgb(mask_idx: np.ndarray, cmap: Dict[int, Tuple[int,int,int]]) -> np.ndarray:
    h, w = mask_idx.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in cmap.items():
        out[mask_idx == cls] = rgb
    return out


def overlay(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img_rgb, 1.0, mask_rgb, alpha, 0)


def add_title(img_rgb: np.ndarray, title: str) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    bar_h = max(28, int(0.06 * h))
    cv2.rectangle(bgr, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(bgr, title, (10, int(bar_h * 0.75)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------
# DATASET
# ------------------------------------------------
class SurgTestDataset(Dataset):
    def __init__(self, items, resize_to, norm_mean, norm_std, cmap):
        self.items = items
        self.resize_to = resize_to
        self.norm_mean = torch.tensor(norm_mean).view(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).view(3, 1, 1)
        self.cmap = cmap

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, msk_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        mask_rgb = np.array(Image.open(msk_path).convert("RGB"), dtype=np.uint8)
        mask_idx = rgb_to_index(mask_rgb, self.cmap)

        new_w, new_h = self.resize_to
        img_np = img.resize((new_w, new_h), resample=Image.BILINEAR)
        mask_np = Image.fromarray(mask_idx).resize((new_w, new_h), resample=Image.NEAREST)

        img_np = np.array(img_np, dtype=np.uint8)
        mask_np = np.array(mask_np, dtype=np.uint8)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - self.norm_mean) / self.norm_std

        return img_t, mask_np, str(img_path), str(msk_path)


# ------------------------------------------------
# MODEL
# ------------------------------------------------
class SegFormerModel(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=n_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return model


# ------------------------------------------------
# METRICS
# ------------------------------------------------
def per_class_iou(pred, gt, n_classes):
    inter = np.zeros(n_classes)
    union = np.zeros(n_classes)
    for k in range(n_classes):
        pk, gk = pred == k, gt == k
        inter[k] = np.logical_and(pk, gk).sum()
        union[k] = np.logical_or(pk, gk).sum()
    return inter, union


def boundary_map(bin_mask):
    if bin_mask.sum() == 0:
        return np.zeros_like(bin_mask, dtype=bool)
    er = binary_erosion(bin_mask, iterations=1)
    return np.logical_xor(bin_mask, er)


def nsd(pred, gt, k, tau):
    pb, gb = boundary_map(pred == k), boundary_map(gt == k)
    if pb.sum() == gb.sum() == 0: return 1.0
    if pb.sum() == 0 or gb.sum() == 0: return 0.0
    dist_gb = distance_transform_edt(~gb)
    dist_pb = distance_transform_edt(~pb)
    return ( (dist_gb[pb] <= tau).sum() + (dist_pb[gb] <= tau).sum() ) / (pb.sum() + gb.sum())


# ------------------------------------------------
# EVALUATION
# ------------------------------------------------
@torch.no_grad()
def predict_tta(model, img_t, device, cfg):
    model.eval()
    aug_inps, inv_ops = [img_t], [lambda y: y]

    deg = cfg["tta"]["rotation_deg"]
    for d in (+deg, -deg):
        aug_inps.append(TF.rotate(img_t, d))
        inv_ops.append(lambda y, dd=d: TF.rotate(y, -dd))

    if cfg["tta"]["flip"]:
        aug_inps.append(TF.hflip(img_t))
        inv_ops.append(lambda y: TF.hflip(y))

    logits_sum = None
    for x, inv in zip(aug_inps, inv_ops):
        x = x.unsqueeze(0).to(device)
        logits = model(x).squeeze(0).cpu()
        logits_inv = inv(logits)
        logits_sum = logits_inv if logits_sum is None else logits_sum + logits_inv

    logits_ens = logits_sum / len(aug_inps)
    return logits_ens.argmax(0).numpy().astype(np.uint8)


def evaluate_folder(model, device, video_dir, cfg):
    print(f"\n[INFO] Evaluating {video_dir.name}")
    pairs = find_pairs(video_dir)
    if not pairs: return

    out_dir = video_dir / "test_results"
    pred_dir = out_dir / "predictions"
    vis_dir = out_dir / "visuals"
    ensure_dir(pred_dir); ensure_dir(vis_dir)

    ds = SurgTestDataset(pairs, cfg["preprocessing"]["resize_to"],
                         cfg["preprocessing"]["norm_mean"],
                         cfg["preprocessing"]["norm_std"],
                         cfg["classes"])
    dl = DataLoader(ds, batch_size=cfg["test"]["batch_size"], shuffle=False,
                    num_workers=cfg["test"]["num_workers"], pin_memory=True)

    inter_sum = np.zeros(len(cfg["classes"]))
    union_sum = np.zeros(len(cfg["classes"]))
    nsd_accum = {k: [] for k in cfg["classes"]}

    for img_t, mask_np, img_path, msk_path in tqdm(dl, desc=video_dir.name):
        img_t = img_t.to(device)
        pred = predict_tta(model, img_t[0].cpu(), device, cfg)

        gt = mask_np[0].numpy().astype(np.uint8)
        stem = Path(img_path[0]).name

        pred_rgb = index_to_rgb(pred, cfg["classes"])
        Image.fromarray(pred).save(pred_dir / stem)

        img_rgb = np.array(Image.open(img_path[0]).convert("RGB"))
        gt_rgb = index_to_rgb(gt, cfg["classes"])
        overlay_img = overlay(img_rgb, pred_rgb)

        grid = np.concatenate([
            add_title(img_rgb, "Original"),
            add_title(gt_rgb, "Ground Truth"),
            add_title(pred_rgb, "Prediction"),
            add_title(overlay_img, "Overlay")], axis=1)
        cv2.imwrite(str(vis_dir / (Path(stem).stem + "_vis.jpg")), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        inter, union = per_class_iou(pred, gt, len(cfg["classes"]))
        inter_sum += inter; union_sum += union

        for k in cfg["classes"]:
            nsd_accum[k].append(nsd(pred, gt, k, cfg["nsd"]["tau_pix"]))

    ious = [inter_sum[k] / union_sum[k] if union_sum[k] > 0 else 1.0 for k in cfg["classes"]]
    nsds = [np.mean(nsd_accum[k]) if nsd_accum[k] else 1.0 for k in cfg["classes"]]

    mIoU, mNSD = np.mean(ious), np.mean(nsds)
    score = np.sqrt(mIoU * mNSD)

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("video,mIoU,mNSD,score,datetime\n")
        f.write(f"{video_dir.name},{mIoU:.6f},{mNSD:.6f},{score:.6f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[INFO] {video_dir.name}: mIoU={mIoU:.3f}, mNSD={mNSD:.3f}, Score={score:.3f}")


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["test"]["seed"])
    device = cfg["test"]["device"]

    model = SegFormerModel(cfg["test"]["segformer_model"], len(cfg["classes"])).to(device)
    model = load_checkpoint(model, Path(cfg["test"]["ckpt_path"]), device)

    root = Path(cfg["test"]["test_root"])
    videos = list_videos(root)

    all_lines = []
    for v in videos:
        evaluate_folder(model, device, v, cfg)
        sfile = v / "test_results" / "summary.csv"
        if sfile.exists():
            with open(sfile, "r") as f:
                lines = f.read().strip().splitlines()
            if len(lines) > 1:
                all_lines.append(lines[1])

    if all_lines:
        out = Path(cfg["output"]["out_dir"]) / "metrics.csv"
        ensure_dir(out.parent)
        with open(out, "w") as f:
            f.write("video,mIoU,mNSD,score,datetime\n")
            for line in all_lines:
                f.write(line + "\n")
        print(f"[INFO] Overall results saved to {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 7: SegFormer Test")
    parser.add_argument("--config", required=True, help="Path to step7_test_segformer.yaml")
    args = parser.parse_args()
    main(args.config)
