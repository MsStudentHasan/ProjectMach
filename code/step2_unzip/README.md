Step 2: Unzip and Reorganize
Purpose
Extracts and reorganizes the SAR-RARP50 dataset archives downloaded in Step 1. Creates a clean directory structure where each video contains organized image frames and corresponding segmentation masks. Removes unnecessary files to optimize storage usage.
Input Requirements
Required Files:
•	data/train.zip - Training dataset archive from Step 1
•	data/test.zip - Test dataset archive from Step 1
Configuration File: config/step2_config.yaml
•	Input and output directory paths
•	Processing parameters
Process Description
The reorganization script performs the following operations:
1.	Extracts both training and test archives
2.	Organizes each video into standardized structure: 
o	segmentation/ - Grayscale PNG mask files
o	images/ - RGB frames extracted from video files
3.	Extracts only frames that have corresponding segmentation masks
4.	Removes unnecessary files: 
o	video_left.avi files
o	action_continuous.txt files
o	action_discrete.txt files
Output
Directory Structure:
output/output_step_02/
├── train/
│   ├── video_01/
│   │   ├── segmentation/     # Grayscale mask files
│   │   └── images/          # RGB frame files
│   ├── video_02/
│   │   ├── segmentation/
│   │   └── images/
│   └── README.md
└── test/
    ├── video_03/
    │   ├── segmentation/
    │   └── images/
    └── README.md
Usage Instructions
1.	Verify Prerequisites: Ensure Step 1 completed successfully with both archives present in data/
2.	Check Configuration: Verify config/step2_config.yaml contains correct project root and path specifications
3.	Execute Processing:
4.	D:\ProjectMach\.venv\Scripts\python.exe D:\ProjectMach\code\step2_unzip\unzip.py
Notes
•	Processing time depends on archive size and system performance
•	Only frames with corresponding segmentation masks are extracted
•	Original archive files remain in data/ directory
•	Progress indicators show extraction status for each video

