"""
evaluation.py
-------------
Compute SSIM, LPIPS, and FID for Virtual Try-On outputs.
Uses class-wise folders to match ground-truth and generated images.
Saves metrics to outputs/metrics.json.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import pil_to_tensor, get_device, list_images, create_dir

# ----------------------------
# Paths
# ----------------------------
GT_DIR = "data/processed/images/test/"         # ground-truth images
GEN_DIR = "outputs/inference_samples/"        # generated images
OUTPUT_JSON = "outputs/metrics.json"

# ----------------------------
# Device
# ----------------------------
device = get_device()
print(f"[INFO] Using device: {device}")

# ----------------------------
# Transforms
# ----------------------------
transform = T.Compose([T.Resize((128, 128))])

# ----------------------------
# Initialize LPIPS
# ----------------------------
lpips_loss = lpips.LPIPS(net='alex').to(device)

# ----------------------------
# Prepare lists for SSIM/LPIPS
# ----------------------------
ssim_scores = []
lpips_scores = []

# Loop over class folders
for cls in sorted(os.listdir(GT_DIR)):
    gt_cls_dir = os.path.join(GT_DIR, cls)
    gen_cls_dir = os.path.join(GEN_DIR, cls)

    if not os.path.isdir(gt_cls_dir):
        continue
    if not os.path.exists(gen_cls_dir):
        print(f"[WARNING] Generated class folder missing: {gen_cls_dir}")
        continue

    gt_images = list_images(gt_cls_dir)
    gen_images = list_images(gen_cls_dir)

    if len(gt_images) != len(gen_images):
        print(f"[WARNING] Mismatch in {cls}: GT={len(gt_images)}, GEN={len(gen_images)}")
        min_len = min(len(gt_images), len(gen_images))
        gt_images = gt_images[:min_len]
        gen_images = gen_images[:min_len]

    for gt_path, gen_path in tqdm(zip(gt_images, gen_images), total=len(gt_images), desc=f"Class {cls}"):
        gt_img = Image.open(gt_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")

        # resize
        gt_img_resized = transform(gt_img)
        gen_img_resized = transform(gen_img)

        # ----------------------------
        # SSIM (grayscale, convert to numpy)
        # ----------------------------
        gt_gray = np.array(gt_img_resized.convert("L"))
        gen_gray = np.array(gen_img_resized.convert("L"))
        ssim_score = ssim(gt_gray, gen_gray)
        ssim_scores.append(ssim_score)

        # ----------------------------
        # LPIPS (tensors in [-1,1])
        # ----------------------------
        gt_tensor = pil_to_tensor(gt_img_resized, size=(128, 128)).to(device) * 2 - 1
        gen_tensor = pil_to_tensor(gen_img_resized, size=(128, 128)).to(device) * 2 - 1
        lpips_score = lpips_loss(gt_tensor, gen_tensor).item()
        lpips_scores.append(lpips_score)

# ----------------------------
# FID calculation
# ----------------------------
try:
    # Compute FID using pytorch-fid
    fid_value = calculate_fid_given_paths(
        [GT_DIR, GEN_DIR],
        batch_size=50,
        device=str(device),
        dims=2048
    )
except Exception as e:
    print(f"[ERROR] FID computation failed: {e}")
    fid_value = None

# ----------------------------
# Save metrics
# ----------------------------
metrics = {
    "SSIM_mean": float(np.mean(ssim_scores)) if ssim_scores else None,
    "LPIPS_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
    "FID": float(fid_value) if fid_value is not None else None
}

create_dir(os.path.dirname(OUTPUT_JSON))
with open(OUTPUT_JSON, "w") as f:
    json.dump(metrics, f, indent=4)

print("[INFO] Evaluation complete. Metrics saved to", OUTPUT_JSON)
print(metrics)
