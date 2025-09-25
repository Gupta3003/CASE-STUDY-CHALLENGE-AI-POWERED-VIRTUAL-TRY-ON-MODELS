"""
train_fast.py
-------------
Faster training for Fashion-MNIST Autoencoder prototype.
Uses smaller images, smaller dataset subset, GPU if available, and fewer epochs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from tqdm import tqdm

from dataset import FashionDataset
from model_viton import Autoencoder

# -------------------- Config --------------------
BATCH_SIZE = 32          # smaller batch for faster training
EPOCHS = 25          # fewer epochs
LR = 1e-3
LATENT_DIM = 128
IMG_SIZE = (128, 128)    # smaller image size for speed

TRAIN_DIR = "data/processed/images/train"
VAL_DIR = "data/processed/images/test"
CHECKPOINT_DIR = "models/checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# -------------------- Data Loaders --------------------
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

train_dataset = FashionDataset(root=TRAIN_DIR, transform=transform)
val_dataset = FashionDataset(root=VAL_DIR, transform=transform)

# Optional: use only a subset for super-fast prototyping
train_dataset = Subset(train_dataset, list(range(min(1000, len(train_dataset)))))
val_dataset = Subset(val_dataset, list(range(min(500, len(val_dataset)))))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# -------------------- Model, Loss, Optimizer --------------------
model = Autoencoder(latent_dim=LATENT_DIM, img_size=IMG_SIZE[0]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------- Training Functions --------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(train_loader.dataset)

def validate():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            recon = model(imgs)
            loss = criterion(recon, imgs)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(val_loader.dataset)

# -------------------- Training Loop --------------------
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(epoch)
    val_loss = validate()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(CHECKPOINT_DIR, f"autoencoder_fast_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Saved checkpoint: {save_path}")

print("[INFO] Fast training complete!")
