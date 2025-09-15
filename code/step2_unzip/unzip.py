#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Unzip and Reorganize
=============================

Purpose:
--------
- Extract train.zip and test.zip from data/
- For each video archive:
    * Extract segmentation/ (grayscale masks)
    * Extract images/ by matching filenames from video_left.avi
    * Discard video_left.avi and action_continuous.txt / action_discrete.txt
- Save results into output/output_step_02/train/ and test/
- Each video folder must contain:
    segmentation/  (PNG grayscale masks)
    images/        (RGB frames)
"""

import os
import zipfile
import yaml
import cv2
import shutil

from pathlib import Path


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def unzip_main_archive(zip_path: Path, extract_to: Path):
    """Unzip the main train/test archive into a temp folder."""
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def extract_video_archives(temp_dir: Path, output_dir: Path):
    """Process each video archive inside temp_dir into output_dir."""
    for item in temp_dir.glob("*.zip"):
        video_id = item.stem
        video_out = output_dir / video_id
        video_out.mkdir(parents=True, exist_ok=True)

        # unzip video archive
        with zipfile.ZipFile(item, "r") as z:
            z.extractall(video_out)

        # keep only segmentation + extracted images
        seg_dir = video_out / "segmentation"
        avi_file = video_out / "video_left.avi"

        # create images folder
        images_dir = video_out / "images"
        images_dir.mkdir(exist_ok=True)

        # extract frames that match mask names
        if avi_file.exists() and seg_dir.exists():
            mask_files = sorted([f.stem for f in seg_dir.glob("*.png")])
            cap = cv2.VideoCapture(str(avi_file))
            frame_idx = 0
            saved = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                fname = f"frame{frame_idx:04d}"
                if fname in mask_files:
                    cv2.imwrite(str(images_dir / f"{fname}.jpg"), frame)
                    saved += 1
                frame_idx += 1
            cap.release()

        # cleanup unwanted files
        if avi_file.exists():
            avi_file.unlink()
        for txt_file in ["action_continuous.txt", "action_discrete.txt"]:
            txt_path = video_out / txt_file
            if txt_path.exists():
                txt_path.unlink()


def main():
    root_dir = Path("D:/ProjectMach")  # adjust if needed
    config_path = root_dir / "config" / "step2_config.yaml"
    cfg = load_config(config_path)

    data_dir = root_dir / "data"
    output_dir = root_dir / "output" / "output_step_02"
    temp_dir = root_dir / "temp_step2"

    # clean temp dir if exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # process both train and test
    for subset in ["train", "test"]:
        archive = data_dir / f"{subset}.zip"
        subset_temp = temp_dir / subset
        subset_temp.mkdir()
        unzip_main_archive(archive, subset_temp)

        subset_out = output_dir / subset
        subset_out.mkdir(parents=True, exist_ok=True)

        extract_video_archives(subset_temp, subset_out)

        # add README
        readme_path = subset_out / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"This folder contains the reorganized {subset} dataset.\n"
                    f"Each video has segmentation/ (grayscale masks) and images/ (RGB frames).")

    # cleanup temp
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
