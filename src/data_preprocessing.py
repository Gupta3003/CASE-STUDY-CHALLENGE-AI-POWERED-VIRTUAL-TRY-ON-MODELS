"""
data_preprocessing.py
---------------------
Converts Fashion-MNIST CSV files (train & test) into image files.
Organizes them into data/processed/images/train/ and test/ directories
with subfolders for each clothing class (0–9).

Windows-safe: Uses absolute paths to avoid FileNotFoundError.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# ----------------------------
# Paths (Windows-safe)
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "viton_hd")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "images")

TRAIN_CSV = os.path.join(RAW_DIR, "fashion-mnist_train.csv")
TEST_CSV = os.path.join(RAW_DIR, "fashion-mnist_test.csv")

TRAIN_OUT = os.path.join(PROCESSED_DIR, "train")
TEST_OUT = os.path.join(PROCESSED_DIR, "test")

IMG_SIZE = (128, 128)  # upscale 28x28 to 128x128

# ----------------------------
# Function to convert CSV → images
# ----------------------------
def save_images_from_csv(csv_file, out_dir):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    print(f"[INFO] Loading {csv_file} ...")
    df = pd.read_csv(csv_file)

    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    # reshape into 28x28
    images = pixels.reshape(-1, 28, 28)

    # save images into class folders
    for i, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images))):
        im = Image.fromarray(img.astype(np.uint8), mode="L")  # grayscale
        im = im.resize(IMG_SIZE, Image.BICUBIC)

        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        fname = f"{os.path.splitext(os.path.basename(csv_file))[0]}_{i}.png"
        im.save(os.path.join(class_dir, fname))

    print(f"[INFO] Saved {len(images)} images into {out_dir}")


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_images_from_csv(TRAIN_CSV, TRAIN_OUT)
    save_images_from_csv(TEST_CSV, TEST_OUT)
