import os

# Step 0: project setup script
# Modify `root_dir` below to the absolute path of your project root before running this script.

# Example: for a Windows environment use a raw string like r"D:\ProjectMach";
# on Unix-like systems, use a path like "/home/user/ProjectMach".
root_dir = r"D:\ProjectMach"

# Define directories relative to the root directory
directories = [
    "config",
    "code/step1_download",
    "code/step2_unzip",
    "code/step3_analyze_rgb",
    "code/step4_split",
    "code/step5_augment",
    "code/step6_train_model",
    "code/step7_test_model",
    "data",
    "output/output_step_01",
    "output/output_step_02/train",
    "output/output_step_02/test",
    "output/output_step_03",
    "output/output_step_03/train",
    "output/output_step_03/test",
    "output/output_step_04",
    "output/output_step_05/train",
    "output/output_step_05/validation",
    "output/output_step_05/test",
    "output/output_step_06/checkpoints",
    "output/output_step_07/predictions",
]

# Define files to create (empty) relative to the root directory
files = [
    "README.md",
    "project_config.yaml",
    # Config files for each step
    "config/step1_config.yaml",
    "config/step2_config.yaml",
    "config/step3_config.yaml",
    "config/step4_config.yaml",
    "config/step5_config.yaml",
    "config/step6_config.yaml",
    "config/step7_config.yaml",
    # Code files and readmes
    "code/step1_download/download.py",
    "code/step1_download/README.md",
    "code/step2_unzip/unzip.py",
    "code/step2_unzip/README.md",
    "code/step3_analyze_rgb/analyze_rgb.py",
    "code/step3_analyze_rgb/README.md",
    "code/step4_split/split.py",
    "code/step4_split/README.md",
    "code/step5_augment/augment.py",
    "code/step5_augment/README.md",
    "code/step6_train_model/train_model.py",
    "code/step6_train_model/README.md",
    "code/step7_test_model/test_model.py",
    "code/step7_test_model/README.md",
    # Placeholder data zip files (empty)
    "data/train.zip",
    "data/test.zip",
    # Output Step 1
    "output/output_step_01/README.md",
    # Output Step 2 placeholders
    "output/output_step_02/train/README.md",
    "output/output_step_02/test/README.md",
    # Output Step 3
    "output/output_step_03/class_distribution.csv",
    "output/output_step_03/train/README.md",
    "output/output_step_03/test/README.md",
    # Output Step 4
    "output/output_step_04/split_assignments.csv",
    # Output Step 5
    "output/output_step_05/train/README.md",
    "output/output_step_05/validation/README.md",
    "output/output_step_05/test/README.md",
    # Output Step 6
    "output/output_step_06/checkpoints/README.md",
    "output/output_step_06/best_model.pt",
    "output/output_step_06/training_metrics.csv",
    "output/output_step_06/training_log.txt",
    # Output Step 7
    "output/output_step_07/predictions/README.md",
    "output/output_step_07/metrics.csv",
]


def create_dirs(root, dir_list):
    for d in dir_list:
        path = os.path.join(root, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def create_files(root, file_list):
    for f in file_list:
        path = os.path.join(root, f)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create empty file if it doesn't exist
        if not os.path.exists(path):
            open(path, 'w').close()
            print(f"Created file: {path}")
        else:
            print(f"File already exists: {path}")

def main():
    print(f"Setting up project structure in: {root_dir}")
    create_dirs(root_dir, directories)
    create_files(root_dir, files)
    print("Project setup complete.")


if __name__ == "__main__":
    main()
