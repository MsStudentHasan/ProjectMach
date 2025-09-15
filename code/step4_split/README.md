Step 4: Dataset Split
Purpose
Creates stratified training and validation splits from the training videos while maintaining proportional class representation. Implements a 90:10 train-validation split using multi-label stratification to ensure balanced class distribution across both subsets.
Input Requirements
Statistical Data: output/output_step_03/class_distribution.csv
•	Per-frame class pixel counts from Step 3 analysis
Dataset Structure:
•	output/output_step_03/train/ - Training video directories with images and masks
•	output/output_step_03/test/ - Test video directories with images and masks
Configuration File: config/step4_config.yaml
•	Split ratios and processing parameters
Process Description
Class Distribution Aggregation:
1.	Aggregates per-class pixel counts for each video from frame-level statistics
2.	Computes video-level class representation metrics
3.	Identifies class distribution patterns across the training set
Multi-Label Stratified Split:
1.	Applies stratification algorithm to maintain class balance
2.	Ensures each anatomical structure class is proportionally represented
3.	Assigns 90% of training videos to training subset
4.	Assigns 10% of training videos to validation subset
5.	Preserves all test videos in the test category
Output
Split Assignment File:
•	output/output_step_04/split_assignments.csv - Maps each video to train, validation, or test subset
File Structure:
video_id,split_assignment
video_01,train
video_02,validation
video_03,test
...
Usage Instructions
Execute from the project root directory:
python code/step4_split/split.py
Alternative with explicit configuration:
python code/step4_split/split.py --config config/step4_config.yaml
Notes
•	Stratification ensures balanced representation of all 10 anatomical classes
•	Test set videos remain unchanged and are not included in train/validation split
•	Split assignments are deterministic and reproducible
•	Class balance verification is performed automatically during split generation
•	The resulting split maintains statistical similarity between train and validation subsets

