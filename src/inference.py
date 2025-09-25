"""
inference.py
------------
Runs the trained Autoencoder on test images.
Saves generated images in class-wise folders under outputs/inference_samples/
matching ground-truth structure for evaluation.
"""

import os
import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FashionDataset
from model_viton import Autoencoder
from utils import get_device, create_dir

# --------------------
# Config
# --------------------
CHECKPOINT_DIR = "models/checkpoints/"
TEST_DIR = "data/processed/images/test"
OUT_DIR = "outputs/inference_samples/"
BATCH_SIZE = 32
LATENT_DIM = 128
DEVICE = get_device()
create_dir(OUT_DIR)

# --------------------
# Automatically pick the latest checkpoint
# --------------------
checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")

# Sort by epoch number (assumes naming: autoencoder_epoch{num}.pth)
checkpoint_files.sort(key=lambda x: int(x.split("epoch")[1].split(".pth")[0]))
CHECKPOINT = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
print(f"[INFO] Using latest checkpoint: {CHECKPOINT}")

# --------------------
# Data Loader
# --------------------
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Custom dataset that returns filename and class
class FashionDatasetWithFilenames(FashionDataset):
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label, os.path.basename(img_path)

test_dataset = FashionDatasetWithFilenames(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------
# Load Model
# --------------------
model = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

print(f"[INFO] Loaded model checkpoint: {CHECKPOINT}")
print(f"[INFO] Running inference on {len(test_dataset)} images")

# --------------------
# Run Inference
# --------------------
with torch.no_grad():
    for imgs, labels, filenames in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(DEVICE)
        recon = model(imgs)

        # Denormalize from [-1,1] to [0,1]
        recon_denorm = (recon * 0.5) + 0.5

        # Save each reconstructed image in class folder
        for i in range(len(filenames)):
            class_folder = os.path.join(OUT_DIR, str(labels[i].item()))
            create_dir(class_folder)
            save_path = os.path.join(class_folder, filenames[i])
            save_image(recon_denorm[i], save_path)

print(f"[INFO] Inference complete. Generated images saved in class-wise folders under {OUT_DIR}")
