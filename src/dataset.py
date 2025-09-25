"""
dataset.py
----------
PyTorch Dataset class for Fashion-MNIST images converted from CSV.
Loads images from data/processed/images/{train,test}/{class}/ directories.

Usage:
    from dataset import FashionDataset
    from torch.utils.data import DataLoader

    train_data = FashionDataset(root="data/processed/images/train", transform=...)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FashionDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Path to data folder (train or test).
                        Example: data/processed/images/train
            transform (callable, optional): torchvision transforms for images.
        """
        self.root = root
        self.transform = transform

        # collect (image_path, label) pairs
        self.samples = []
        for label in sorted(os.listdir(root)):
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(label_dir, fname), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


# Example test run
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    # basic transforms: resize → tensor → normalize
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))  # for grayscale images
    ])

    dataset = FashionDataset(root="data/processed/images/train", transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"[INFO] Dataset size: {len(dataset)} images")

    # test iterate
    imgs, labels = next(iter(loader))
    print("[INFO] Batch shape:", imgs.shape)
    print("[INFO] Labels:", labels)
