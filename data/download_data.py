import os
import subprocess

# Set the path where you want to download the dataset
download_path = "../Grounded-SAM-2/data"

# Ensure the directory exists
os.makedirs(download_path, exist_ok=True)

# Kaggle dataset URL or identifier
dataset = "rohanmallick/kinetics-train-5per"

# Kaggle download command
download_command = f"kaggle datasets download -d {dataset} -p {download_path} --unzip"

# Execute the download command
subprocess.run(download_command, shell=True, check=True)

print(f"Dataset downloaded to {download_path}")
