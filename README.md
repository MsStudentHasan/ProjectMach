# SAR-RARP50 Surgical Instrument Segmentation Pipeline

## Project Overview

This repository implements a **comprehensive, reproducible pipeline** for semantic segmentation of surgical instruments using the **SAR-RARP50 dataset**.
The project supports both **Windows** and **Unix/Linux** environments, with modular **code**, **configuration files**, and **outputs** organized by processing step.

* **Dataset**: 50 robotic prostatectomy (RARP) suturing videos with pixel-wise annotations
* **Objective**: Train and evaluate robust segmentation models for surgical tool understanding
* **Scope**: End-to-end workflow, from **data acquisition** to **model evaluation**

---

Run each step from 0 to 7 after setting up your configurations in config files, everything will be done automatically.
---
## Directory Structure

After running **Step 0 (Setup)**, your project root will look like this:

```
ProjectMach/
├── config/                # YAML configuration files (step1_config.yaml ... step7_config.yaml)
├── code/                  # Scripts per processing step
│   ├── step1_download/
│   ├── step2_unzip/
│   ├── step3_analyze_rgb/
│   ├── step4_split/
│   ├── step5_augment/
│   ├── step6_train_model/
│   └── step7_test_model/
├── data/                  # Raw dataset archives (train.zip, test.zip)
├── output/                # Outputs organized by step (01–07)
├── project_config.yaml    # Global config with root paths
└── README.md              # This file
```

### Output Structure

Each step generates its own structured results:

```
output/
├── output_step_01/   # Download confirmation
├── output_step_02/   # Extracted train/test (images + segmentation)
├── output_step_03/   # RGB masks + class distribution CSV
├── output_step_04/   # Dataset split assignments
├── output_step_05/   # Augmented train/validation/test sets
├── output_step_06/   # Model checkpoints, logs, metrics
└── output_step_07/   # Predictions + evaluation results
```

---

## Pipeline Steps

1. **Step 0 – Setup**
   Initialize directory structure and placeholder configs.

2. **Step 1 – Download**
   Download training/test archives into `data/`.
   Links:

   * [Train set](https://rdr.ucl.ac.uk/ndownloader/articles/24932529/versions/1)
   * [Test set](https://rdr.ucl.ac.uk/ndownloader/articles/24932499/versions/1)

3. **Step 2 – Unzip & Reorganize**
   Extract and reorganize into `images/` + `segmentation/` folders.

4. **Step 3 – Analyze & RGB Masks**

   * Compute per-class pixel statistics.
   * Convert grayscale masks → RGB masks.

5. **Step 4 – Train/Validation Split**

   * Perform **90:10 stratified split** (preserving class balance).
   * Test set remains unchanged.

6. **Step 5 – Augmentation**

   * Apply **class-aware augmentations** (rotation, flip, zoom, brightness, blur).
   * Address class imbalance.

7. **Step 6 – Train Model**

   * Train a **SegFormer model** on 960×540 frames.
   * Save checkpoints + logs.

8. **Step 7 – Evaluation**

   * Run inference on test set.
   * Compute metrics and export predictions.

---

## Requirements

* **System**:

  * Python 3.10+
  * CUDA-enabled GPU (recommended)

* **Dependencies**:

  ```
  torch, torchvision
  tqdm, numpy, pandas
  Pillow, OpenCV, matplotlib
  pyyaml
  ```

---

## Quick Start

1. **Initialize Project**

   ```bash
   python code/step0_setup.py
   ```

2. **Download Dataset**

   ```bash
   python code/step1_download/download.py --config config/step1_config.yaml
   ```

3. **Run Remaining Steps**
   Execute steps 2–7 sequentially, each with its own config YAML.

---

## Evaluation Metrics

* **Mean IoU (mIoU)**: Pixel-level overlap
* **Mean Normalized Surface Dice (mNSD)**: Boundary-sensitive metric
* **Segmentation Score**:

  $$
  Score = \sqrt{mIoU \times mNSD}
  $$

---

## References

* **Dataset**: [SAR-RARP50 Challenge](https://rdr.ucl.ac.uk/projects/SAR-RARP50_Segmentation_of_surgical_instrumentation_and_Action_Recognition_on_Robot-Assisted_Radical_Prostatectomy_Challenge/191091)
* **Pipeline Guide**: Complete Updated Project Documentation
* **Challenge Paper**: *Psychogyios et al., EndoVis 2022 SAR-RARP50*


