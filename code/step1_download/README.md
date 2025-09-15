Step 1: Download Data
Purpose
Downloads the SAR-RARP50 dataset archives (training and test sets) to the local data/ directory. This step ensures both required ZIP files are available for subsequent processing steps.
Input Requirements
Configuration File: config/step1_config.yaml
•	train_url: Download URL for training dataset archive
•	test_url: Download URL for test dataset archive
Global Configuration: project_config.yaml
•	Root directory path definitions
Process Description
The download script performs the following operations:
1.	Reads dataset URLs from the step configuration file
2.	Downloads archives to the designated data directory: 
o	data/train.zip
o	data/test.zip
3.	Generates confirmation documentation in the output directory
Output
Downloaded Files:
•	data/train.zip - Training dataset archive
•	data/test.zip - Test dataset archive
Documentation:
•	output/output_step_01/README.md - Download confirmation and status
Usage Instructions
Execute from the project root directory:
D:\ProjectMach\.venv\Scripts\python.exe D:\ProjectMach\code\step1_download\download.py
Alternative with explicit configuration:
python code/step1_download/download.py --config config/step1_config.yaml
Notes
•	Ensure internet connectivity before running this step
•	Download progress will be displayed during execution
•	Both archives must be successfully downloaded before proceeding to Step 2

