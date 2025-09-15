Step 6: Model Training
Purpose
Trains a SegFormer model for semantic segmentation of surgical instruments using the augmented dataset from Step 5. Implements comprehensive training with validation monitoring, checkpoint management, and multi-metric evaluation including boundary-sensitive metrics.
Input Requirements
Training Data: output/output_step_05/train/
•	Augmented training images and RGB segmentation masks
Validation Data: output/output_step_05/validation/
•	Augmented validation images and RGB segmentation masks
Configuration File: config/step6_SFconfig.yaml
•	Model architecture, training hyperparameters, and path specifications
Optional Resume Checkpoint:
•	output/output_step_06/checkpoints/epoch_XX.pth - Model-only checkpoint
•	output/output_step_06/segformer/<run>/last.pth - Full training state
•	output/output_step_06/segformer/<run>/best.pth - Best model checkpoint
Process Description
Data Processing:
1.	Loads training and validation splits with stratified sampling
2.	Resizes training images and masks to 960×540 resolution
3.	Maintains original resolution for validation to preserve evaluation accuracy
4.	Applies standard augmentations (random flip and rotation)
Model Training:
1.	Initializes SegFormer architecture for 10-class semantic segmentation
2.	Implements mixed precision training for memory efficiency
3.	Uses adaptive learning rate scheduling
4.	Monitors training progress with comprehensive validation metrics
Evaluation Metrics:
•	Mean IoU (mIoU): Intersection-over-Union for pixel-wise accuracy
•	Mean Normalized Surface Dice (mNSD): Boundary-sensitive segmentation quality
•	Combined Score: √(mIoU × mNSD) for holistic performance assessment
Advanced Validation:
•	Full-resolution validation frame evaluation
•	Test-Time Augmentation (TTA) with flips and rotations
•	Per-class and overall metric computation
Output
Training Results: output/output_step_06/segformer/<YYYYMMDD_HHMMSS>/
Checkpoint Files:
•	last.pth - Complete training state (model, optimizer, scaler)
•	best.pth - Best performing model by combined score
•	epoch_XX.pth - Individual epoch model snapshots
Training Documentation:
•	log.csv - Per-epoch metrics (loss, mIoU, mNSD, combined score)
•	config_effective.json - Frozen configuration used for training run
Global Checkpoints: output/output_step_06/checkpoints/
•	Manual checkpoint storage for training resumption
Usage Instructions
Standard Training:
D:\ProjectMach\.venv\Scripts\python.exe ^
  D:\ProjectMach\code\step6_train_model\step6_SFtrain.py ^
  --config D:\ProjectMach\config\step6_SFconfig.yaml
Resume from Checkpoint:
python code/step6_train_model/step6_SFtrain.py ^
  --config config/step6_SFconfig.yaml ^
  --resume output/output_step_06/checkpoints/epoch_35.pth
Notes
•	Training automatically detects and utilizes CUDA acceleration when available
•	Checkpoint saving occurs after each epoch with automatic cleanup of older files
•	Validation evaluation uses full-resolution images for accurate performance assessment
•	Mixed precision training reduces memory requirements while maintaining numerical stability
•	Training can be resumed from any saved checkpoint without data loss
•	Best model selection is based on the combined score metric for balanced performance

