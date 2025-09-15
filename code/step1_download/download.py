#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Download Data
=====================

Objective:
----------
Download the SAR-RARP50 training and test archives into the data/ folder.

Input:
------
- Dataset URLs from step1_config.yaml

Output:
-------
- data/train.zip
- data/test.zip
- output/output_step_01/README.md updated with confirmation
"""

import os
import yaml
import requests

def download_file(url, dest_path):
    """Download file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded: {dest_path}")

def main():
    # Get root directory from project_config.yaml
    root_config = os.path.join(os.path.dirname(__file__), "..", "..", "project_config.yaml")
    with open(root_config, "r") as f:
        project_cfg = yaml.safe_load(f)
    root_dir = project_cfg["root_dir"]

    # Load step1 configuration
    config_path = os.path.join(root_dir, "config", "step1_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = os.path.join(root_dir, "data")
    output_readme = os.path.join(root_dir, "output", "output_step_01", "README.md")

    os.makedirs(data_dir, exist_ok=True)

    # Download train.zip
    train_zip = os.path.join(data_dir, "train.zip")
    if not os.path.exists(train_zip):
        download_file(cfg["train_url"], train_zip)
    else:
        print("train.zip already exists, skipping download.")

    # Download test.zip
    test_zip = os.path.join(data_dir, "test.zip")
    if not os.path.exists(test_zip):
        download_file(cfg["test_url"], test_zip)
    else:
        print("test.zip already exists, skipping download.")

    # Update README in output_step_01
    with open(output_readme, "w") as f:
        f.write("Step 1 completed successfully.\n")
        f.write(f"Training set downloaded to: {train_zip}\n")
        f.write(f"Test set downloaded to: {test_zip}\n")

    print("Step 1 finished. Files saved in data/ and confirmation written to output_step_01/README.md")

if __name__ == "__main__":
    main()
