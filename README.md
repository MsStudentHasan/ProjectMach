SAR-RARP50 Surgical Instrument Segmentation Pipeline
Project Overview
This repository implements a comprehensive, reproducible pipeline for semantic segmentation of surgical instruments using the SAR-RARP50 dataset. The project is designed for Windows and Unix systems, with modular code, configurations, and outputs organized by processing step.
Dataset: 50 robotic prostatectomy (RARP) suturing videos with pixel-wise annotations
Objective: Train and evaluate robust segmentation models for surgical tool understanding
Scope: Complete workflow from data acquisition to model evaluation
Directory Structure
After running Step 0 (setup), your project directory will contain:
ProjectMach/
├── config/                  # YAML configuration files per step
├── code/                    # Scripts organized by processing step
│   ├── step1_download/
│   ├── step2_unzip/
│   ├── step3_analyze_rgb/
│   ├── step4_split/
│   ├── step5_augment/
│   ├── step6_train_model/
│   └── step7_test_model/
├── data/                    # Raw dataset files (train.zip, test.zip)
├── output/                  # Processing outputs organized by step (01-07)
├── project_config.yaml      # Global configuration (root paths)
└── README.md               # This file
Each processing step generates structured outputs in dedicated folders:
output/
├── output_step_01/     # Download confirmation
├── output_step_02/     # Extracted train/test (images + segmentation)
├── output_step_03/     # RGB masks + class distribution analysis
├── output_step_04/     # Dataset split assignments
├── output_step_05/     # Augmented train/validation/test sets
├── output_step_06/     # Model checkpoints and training logs
└── output_step_07/     # Predictions and evaluation metrics
Pipeline Steps
Step 0 - Setup: Creates directory structure, placeholder configurations, and initialization files
Step 1 - Download: Downloads training and test archives to data/ directory
Step 2 - Unzip & Reorganize: Extracts videos and organizes into images/ and segmentation/ folders
Step 3 - Analyze & RGB Masks: Computes class distributions and converts grayscale to RGB masks
Step 4 - Train/Val Split: Creates stratified 90:10 split preserving class distributions
Step 5 - Augmentation: Applies class-aware data augmentation strategies
Step 6 - Train Model: Trains SegFormer model with 960×540 resolution and saves checkpoints
Step 7 - Evaluation: Evaluates model on test set and computes comprehensive metrics
Requirements
System Requirements:
•	Python 3.10 or higher
•	PyTorch with CUDA support (recommended)
Dependencies:
•	tqdm, numpy, pandas, Pillow, OpenCV, matplotlib, pyyaml
Quick Start
1.	Initialize Project Structure:
2.	python code/step0_setup.py
3.	Download Dataset:
4.	python code/step1_download/download.py --config config/step1_config.yaml
5.	Execute Remaining Steps: Progress sequentially through Steps 2-7, each using its respective configuration file.
Evaluation Metrics
Mean IoU (mIoU): Measures pixel-wise overlap between predictions and ground truth
Mean Normalized Surface Dice (mNSD): Evaluates boundary-sensitive segmentation quality
Segmentation Score: Combined metric calculated as √(mIoU × mNSD)
References
Dataset Source: SAR-RARP50 Challenge
Pipeline Documentation: Complete Updated Guide for SAR-RARP50

