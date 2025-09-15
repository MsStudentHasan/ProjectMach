#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Analyze Masks & Create RGB Masks
========================================

- Computes per-class pixel distribution for all masks.
- Converts grayscale masks to RGB masks using the predefined COLOR_MAP.
- Skips RGB conversion if RGB_segmentation folder already exists with files.

Inputs:
- Directories from Step 2: train/ and test/ with segmentation/ and images/

Outputs:
- class_distribution.csv in output_step_03
- train/ and test/ copies with added RGB_segmentation/ folders
"""

import os
import yaml
import csv
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

# Predefined color map (class_id: [R, G, B])
COLOR_MAP = {
    0: [0, 0, 0],       # Background - Black
    1: [255, 0, 0],     # Tool shaft - Red
    2: [0, 255, 0],     # Tool clasper - Green
    3: [0, 0, 255],     # Tool wrist - Blue
    4: [255, 255, 0],   # Thread - Yellow
    5: [255, 0, 255],   # Clamps - Magenta
    6: [0, 255, 255],   # Suturing needle - Cyan
    7: [128, 128, 128], # Suction tool - Gray
    8: [255, 165, 0],   # Catheter - Orange
    9: [128, 0, 128],   # Needle Holder - Purple
}


def load_config():
    # project_root = D:\ProjectMach
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(project_root, "config", "step3_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)




def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_class_distribution(mask_path, num_classes=10):
    mask = np.array(Image.open(mask_path))
    return [(mask == c).sum() for c in range(num_classes)]


def convert_mask_to_rgb(mask_path, save_path):
    mask = np.array(Image.open(mask_path))
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in COLOR_MAP.items():
        rgb_mask[mask == c] = color
    Image.fromarray(rgb_mask).save(save_path)


def process_split(split_name, input_dir, output_dir, writer):
    ensure_dir(output_dir)

    for video_id in tqdm(os.listdir(input_dir), desc=f"Processing {split_name}"):
        video_in = os.path.join(input_dir, video_id)
        video_out = os.path.join(output_dir, video_id)

        if not os.path.exists(video_out):
            shutil.copytree(video_in, video_out)

        seg_dir = os.path.join(video_out, "segmentation")
        rgb_dir = os.path.join(video_out, "RGB_segmentation")
        ensure_dir(rgb_dir)

        # Check if RGB already exists
        rgb_exists = os.path.exists(rgb_dir) and len(os.listdir(rgb_dir)) > 0

        for mask_file in os.listdir(seg_dir):
            if not mask_file.lower().endswith(".png"):
                continue
            mask_path = os.path.join(seg_dir, mask_file)

            # 1. Always compute class distribution
            counts = compute_class_distribution(mask_path)
            row = {"video_id": video_id, "image_id": mask_file}
            row.update({f"class_{i}": counts[i] for i in range(10)})
            writer.writerow(row)

            # 2. Only create RGB if it doesn’t exist already
            if not rgb_exists:
                rgb_path = os.path.join(rgb_dir, mask_file)
                convert_mask_to_rgb(mask_path, rgb_path)



def main():
    cfg = load_config()

    csv_path = cfg["output_csv"]
    ensure_dir(os.path.dirname(csv_path))

    fieldnames = ["video_id", "image_id"] + [f"class_{i}" for i in range(10)]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        process_split("train", cfg["input_train_dir"], cfg["output_train_dir"], writer)
        process_split("test", cfg["input_test_dir"], cfg["output_test_dir"], writer)


if __name__ == "__main__":
    main()
