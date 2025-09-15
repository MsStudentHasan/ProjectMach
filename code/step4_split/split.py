#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Train–Validation Split (Proper Multi-Label Stratification)
==================================================================

Objective:
----------
Split only the TRAINING data into train and validation (90:10),
preserving class distribution across all classes.
Test data remains untouched.

Input:
------
- output/output_step_03/class_distribution.csv
- output/output_step_03/train/ and output/output_step_03/test/

Output:
-------
- output/output_step_04/split_assignments.csv
"""

import os
import yaml
import pandas as pd
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # --- Load config ---
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = os.path.join(root_dir, "config", "step4_config.yaml")
    cfg = load_config(cfg_path)

    # Paths from config
    class_csv = cfg["input_class_distribution"]
    step3_train = cfg["input_step3_train"]
    step3_test = cfg["input_step3_test"]
    step4_dir = cfg["output_step4_dir"]
    out_csv = cfg["output_split_assignments"]

    os.makedirs(step4_dir, exist_ok=True)

    # --- Read class distribution ---
    df = pd.read_csv(class_csv)

    # Get list of train/test video IDs from folder structure
    train_videos_list = os.listdir(step3_train)
    test_videos_list = os.listdir(step3_test)

    # Aggregate per video
    agg_df = df.groupby("video_id").sum(numeric_only=True).reset_index()

    # Select only train videos
    agg_train = agg_df[agg_df["video_id"].isin(train_videos_list)]

    # Build X (video_ids) and y (multi-label presence matrix)
    X = agg_train["video_id"].values
    class_cols = [c for c in agg_train.columns if c.startswith("class_")]
    y = (agg_train[class_cols] > 0).astype(int).values

    # --- Stratified split on TRAIN only ---
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(msss.split(X, y))

    train_split = set(X[train_idx])
    val_split = set(X[val_idx])

    # --- Build assignment table ---
    assignments = []

    # Train videos → assign train/validation
    for vid in tqdm(train_videos_list, desc="Assigning train/val"):
        if vid in train_split:
            split = "train"
        elif vid in val_split:
            split = "validation"
        else:
            raise RuntimeError(f"Video {vid} not assigned to train or validation!")
        assignments.append({"video_id": vid, "split": split})

    # Test videos → assign test
    for vid in tqdm(test_videos_list, desc="Assigning test"):
        assignments.append({"video_id": vid, "split": "test"})

    # Save CSV
    assign_df = pd.DataFrame(assignments)
    assign_df.to_csv(out_csv, index=False)

    print(f"[✔] Split assignments saved to {out_csv}")


if __name__ == "__main__":
    main()
