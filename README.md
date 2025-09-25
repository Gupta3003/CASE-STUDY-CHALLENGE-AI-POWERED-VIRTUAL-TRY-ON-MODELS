# Virtual Try-On Project

A deep learning project for **Virtual Try-On** using **Fashion-MNIST** and **Viton-HD** datasets.  
This repository includes data preprocessing, model training, inference, evaluation, and visualization of generated outputs.

---
## Project Report
[Project Report.pdf](https://github.com/user-attachments/files/22536992/Project.Report.pdf)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Used and Why](#dataset-used-and-why)
- [Model Architecture and Reasoning](#model-architecture-and-reasoning)
- [Training Setup & Server Details](#training-setup--server-details)
- [Accuracy / Metrics Achieved](#accuracy--metrics-achieved)
- [Visual Examples](#visual-examples)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
The Virtual Try-On system implemented in this project can:

- Preprocess clothing and person images (masks, keypoints, and splits).  
- Train deep learning models (`viton_baseline` and `hr_vton_experiment1`) for image generation.  
- Perform inference to generate virtual try-on samples.  
- Evaluate generated images using metrics such as **SSIM**, **LPIPS**, and **FID**.  
- Display results through a web interface (`index.html` and `result.html`).

---

## Dataset Used and Why
- **Fashion-MNIST**: Used for initial experiments due to its simplicity and availability.  
- **Viton-HD**: High-resolution dataset of real clothing and human images for accurate try-on results.  

> These datasets provide both variety and high-quality images, which are essential for realistic virtual try-on.

---

## Model Architecture and Reasoning
- **VITON Baseline**: Standard virtual try-on architecture for basic garment transfer.  
- **HR-VTON**: High-resolution model to improve texture, details, and alignment for realistic results.  

> Models were chosen to balance performance and generation quality while supporting both low- and high-resolution images.

---

## Training Setup & Server Details
- **Environment**: Python 3.10, PyTorch 2.x  
- **GPU**: NVIDIA RTX 3090 / CUDA 11.x  
- **Training Hyperparameters**:  
  - Batch size: 8–16  
  - Learning rate: 0.0001  
  - Epochs: 50–100  
- **Data preprocessing**: Images resized to 256×256, normalized, and masks/keypoints extracted.  

---

## Accuracy / Metrics Achieved
- **SSIM**: Structural Similarity Index (measures similarity between generated and ground truth images)  
- **LPIPS**: Learned Perceptual Image Patch Similarity (perceptual similarity)  
- **FID**: Fréchet Inception Distance (distribution distance between generated and real images)  

> Metrics are stored in `outputs/metrics.json`.

---

## Visual Examples
**Input Image + Generated Try-On Output**:

| Front Page | Result Page |
|-------------|-----------------|
| <img width="1366" height="768" alt="Screenshot (1194)" src="https://github.com/user-attachments/assets/fd6b09af-30e5-40b7-927d-5f58f00e3223" /> | <img width="1366" height="768" alt="Screenshot (1195)" src="https://github.com/user-attachments/assets/3078939a-0130-4d9a-8dfd-7f17b9e22bc9" /> |

> Replace the images with your own example files in `static/uploads/` and `outputs/inference_samples/`.

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Gupta3003/virtual-tryon-project.git
cd virtual-tryon-project
```

2. (Optional) Set up a virtual environment:
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Git LFS to manage large files:
```bash
git lfs install
git lfs pull
```

---

## Usage
### Data Preprocessing
```bash
python src/data_preprocessing.py
```

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/inference.py --model models/hr_vton_experiment1.pth --input data/processed/images/test/
```

### Evaluation
```bash
python src/evaluation.py --generated outputs/inference_samples/ --ground_truth data/processed/images/test/
```

### Web Interface
- Open `templates/index.html` in a browser to upload images and view virtual try-on results.

---

## Folder Structure
```
virtual-tryon-project/
├── data/
├── models/
├── outputs/
├── src/
├── templates/
├── static/
├── notebooks/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Contributing
Contributions are welcome! Open an issue or submit a pull request for improvements.

---

## License
This project is licensed under the MIT License.
