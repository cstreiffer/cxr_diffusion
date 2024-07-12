from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

def apply_transformations(image_path, output_path):
    # Load the image using PIL
    pil_image = Image.open(image_path)

    # Resize image to 224x224 using bicubic interpolation
    resized_image = pil_image.resize((224, 224), Image.BICUBIC)

    # Convert PIL image to a numpy array for OpenCV processing
    cv_image = np.array(resized_image)

    # Convert RGB to grayscale (if the image is not already in grayscale)
    if len(cv_image.shape) == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(cv_image)

    # Convert back to PIL image to save as output
    output_image = Image.fromarray(enhanced_image)
    output_image.save(output_path)

import tarfile
import os
import glob
from datetime import datetime

def compress_directory(directory_path, output_path):
    """
    Compresses a directory into a .tar.gz file.

    Args:
    directory_path (str): The path to the directory to compress.
    output_path (str): The path where the compressed file will be saved.
    """
    with tarfile.open(output_path, "w:gz") as tar:
        # Add directory to tar.gz
        tar.add(directory_path, arcname=os.path.basename(directory_path))

    print(f"Directory '{directory_path}' compressed to '{output_path}'")

import tarfile
def decompress_tar_gz(tar_path, output_directory):
    """
    Decompresses a .tar.gz file into a specified output directory.

    Args:
    tar_path (str): The path to the .tar.gz file to decompress.
    output_directory (str): The directory where the contents will be extracted.
    """
    # Open the tar.gz file
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract all contents into the output directory
        tar.extractall(path=output_directory)

    print(f"Contents of '{tar_path}' extracted to '{output_directory}'")

import os
import shutil
def clear_directory(directory_path):
    """
    Removes all files and subdirectories in a given directory.

    Args:
    directory_path (str): The path to the directory to clear.
    """
    # Check each item in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)  # Full path to the item

        # Check if it's a file or directory
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # Remove the file or link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the directory and all its contents

    print(f"All files and directories within '{directory_path}' have been removed.")

# Create the script to run in parallel
import concurrent.futures
from datetime import datetime
import os

# Define variables
def load_all_files():
  img_base_dir = "drive/MyDrive/cxr/data/images/"
  img_dir = "images"
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  # Define the download_data function
  def load_data(f):
    print(f"{datetime.now()} - Now uncompressing {f}")
    decompress_tar_gz(os.path.join(img_base_dir, f"{f}.tar.gz"), './')
    shutil.move(f, f"images/{f}")
    print(f"{datetime.now()} - Finished uncompressing {f}")

  # Define the function to run download_data in parallel
  def run_parallel(files, max_workers=4):
      # Use ThreadPoolExecutor for IO-bound tasks, use ProcessPoolExecutor for CPU-bound tasks
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          # Map the download_data function to the list of files
          results = list(executor.map(load_data, files))
      return results

  def run_sequential(files):
    for f in files:
      load_data(f)

  # Load the files
  files = ["chestxray_nicc", "chexchonet", "chexpert", "mimic"]

  # Run the parallel downloads
  results = run_parallel(files, max_workers=4)