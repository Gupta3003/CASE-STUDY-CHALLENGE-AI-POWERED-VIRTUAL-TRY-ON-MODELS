# helper functions 
import os
from PIL import Image
import torch
from torchvision import transforms

# ----------------------------
# Image / Tensor helpers
# ----------------------------
def load_image(image_path, size=(256, 192)):
    """Load an image and resize."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def save_image(tensor, save_path):
    """Convert a tensor to PIL image and save."""
    img = transforms.ToPILImage()(tensor.squeeze().cpu())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def tensor_to_pil(tensor):
    """Convert a tensor to PIL image."""
    return transforms.ToPILImage()(tensor.squeeze().cpu())

def pil_to_tensor(pil_img, size=(256, 192)):
    """Convert PIL image to tensor."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(pil_img).unsqueeze(0)

# ----------------------------
# Directory / file helpers
# ----------------------------
def create_dir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def list_images(folder_path, exts=(".png", ".jpg", ".jpeg")):
    """Return a sorted list of image file paths in a folder."""
    return sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(exts)
    ])

# ----------------------------
# Device helpers
# ----------------------------
def get_device():
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Debug / visualization helpers
# ----------------------------
def show_image(tensor):
    """Display a tensor image."""
    pil_img = transforms.ToPILImage()(tensor.squeeze().cpu())
    pil_img.show()
