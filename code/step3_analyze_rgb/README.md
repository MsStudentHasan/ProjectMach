Step 3: Analyze Masks and Create RGB Masks
Purpose
Performs comprehensive analysis of segmentation masks and creates color-coded RGB representations. This step generates class distribution statistics and converts grayscale masks to RGB format for improved visualization and downstream processing.
Input Requirements
Data Source: Output from Step 2
•	output/output_step_02/train/<video_id>/segmentation/*.png
•	output/output_step_02/test/<video_id>/segmentation/*.png
Configuration File: config/step3_config.yaml
•	Input/output directory paths
•	Processing parameters
Process Description
Class Distribution Analysis:
1.	Analyzes grayscale segmentation masks from all videos
2.	Counts pixels belonging to each class (0-9) for every frame
3.	Records comprehensive statistics with video and frame identifiers
RGB Mask Generation:
1.	Converts grayscale masks to color-coded RGB representations
2.	Applies predefined color mapping for each class
3.	Saves RGB masks in dedicated RGB_segmentation/ directories
4.	Skips conversion if RGB masks already exist to prevent recomputation
Output
Statistical Analysis:
•	output/output_step_03/class_distribution.csv - Per-frame class pixel counts with columns: video_id, image_id, class_0, class_1, ..., class_9
Processed Dataset:
output/output_step_03/
├── train/<video_id>/
│   ├── segmentation/         # Original grayscale masks (copied)
│   ├── images/              # Original RGB frames (copied)
│   └── RGB_segmentation/    # Color-coded RGB masks (new)
└── test/<video_id>/
    ├── segmentation/
    ├── images/
    └── RGB_segmentation/
Class Color Mapping
Class ID	Anatomical Structure	RGB Color
0	Background	Black (0,0,0)
1	Tool shaft	Red (255,0,0)
2	Tool clasper	Green (0,255,0)
3	Tool wrist	Blue (0,0,255)
4	Thread	Yellow (255,255,0)
5	Clamps	Magenta (255,0,255)
6	Suturing needle	Cyan (0,255,255)
7	Suction tool	Gray (128,128,128)
8	Catheter	Orange (255,165,0)
9	Needle Holder	Purple (128,0,128)
Usage Instructions
Execute from the project root directory:
cd D:\ProjectMach
.venv\Scripts\python.exe code\step3_analyze_rgb\analyze_rgb.py
Alternative with explicit configuration:
python code/step3_analyze_rgb/analyze_rgb.py --config config/step3_config.yaml
Notes
•	RGB mask generation is automatically skipped if output already exists
•	Class distribution analysis runs for all available frames
•	Color mapping is consistent across all processing steps
•	Progress indicators show analysis status for each video

