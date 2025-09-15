Step 5: Data Augmentation
Purpose
Expands the training and validation datasets using class-aware augmentation strategies that focus on underrepresented anatomical structures. Applies intelligent augmentation rules based on class frequency analysis to improve model performance on rare classes while preserving the test set unchanged.
Input Requirements
Statistical Data:
•	output/output_step_03/class_distribution.csv - Per-frame class distributions
•	output/output_step_04/split_assignments.csv - Train/validation/test assignments
Dataset Structure:
•	output/output_step_03/train/ - Training images and RGB masks
•	output/output_step_03/test/ - Test images and RGB masks
Configuration File: config/step5_config.yaml
•	Augmentation parameters and output paths
Process Description
Class Frequency Analysis:
1.	Computes per-video statistics for appearance counts and average pixel densities
2.	Identifies the three least frequent classes across the dataset
3.	Determines augmentation strategy based on class representation thresholds
Intelligent Augmentation Rules:
•	ALL5 Strategy: Applies all five augmentation techniques when frames contain the two rarest classes above average pixel density
•	DOUBLE2 Strategy: Applies two random augmentations when frames contain the third rarest class above average
•	SINGLE1 Strategy: Applies single random augmentation for all other frames
Available Augmentation Techniques:
•	Rotation (±15 degrees)
•	Brightness adjustment (±10%)
•	Zoom scaling (±10%)
•	Gaussian blur (radius=5 pixels)
•	Horizontal flip transformation
Output
Augmented Dataset Structure:
output/output_step_05/
├── train/<video_id>/
│   ├── images/              # Augmented RGB frames
│   └── RGB_segmentation/    # Corresponding augmented masks
├── validation/<video_id>/
│   ├── images/              # Augmented RGB frames
│   └── RGB_segmentation/    # Corresponding augmented masks
└── test/                    # Unchanged test set (direct copy)
Performance Features
Hardware Acceleration:
•	CUDA acceleration support via PyTorch/TorchVision when available
•	Automatic GPU detection and utilization
Parallel Processing:
•	Multi-worker support for faster I/O operations
•	Concurrent augmentation processing where supported
Progress Monitoring:
•	Real-time progress bars showing augmentation type per frame
•	Processing speed and estimated completion time
Usage Instructions
Execute from the project root directory:
python code/step5_augment/augment.py
Alternative with explicit configuration:
python code/step5_augment/augment.py --config config/step5_config.yaml
Notes
•	Test set remains completely unchanged to ensure unbiased evaluation
•	Augmentation strategies adapt automatically based on class distribution analysis
•	Progress indicators display the specific augmentation rule applied to each frame
•	All augmentations maintain spatial correspondence between images and masks
•	CUDA acceleration significantly improves processing speed when available

