# Virtual Try-On Project

A deep learning project for **Virtual Try-On** using **Fashion-MNIST** and **Viton-HD** datasets.  
This repository includes data preprocessing, model training, inference, evaluation, and visualization of generated outputs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Evaluation Metrics](#evaluation-metrics)
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

## Dataset
- **Fashion-MNIST**: Training and testing CSV files located in `data/raw/viton_hd/`.  
- **Viton-HD**: High-resolution images used for virtual try-on experiments.  

> **Note:** Large dataset files (>100 MB) are not included due to GitHub limits. Please download them separately and place them in `data/raw/viton_hd/`.

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
│   ├── raw/
│   │   ├── viton_hd/
│   │   │   ├── fashion_mnist_train.csv
│   │   │   ├── fashion_mnist_test.csv
│   │   └── README.md
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/0/, 1/, ...
│   │   │   ├── test/0/, 1/, ...
│   │   ├── masks/
│   │   ├── keypoints/
│   │   └── splits/
│   └── README.md
├── models/
│   ├── checkpoints/
│   ├── viton_baseline.pth
│   └── hr_vton_experiment1.pth
├── outputs/
│   ├── inference_samples/
│   ├── metrics.json
│   └── logs/
├── src/
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── model_viton.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── uploads/
│   └── styles.css
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── training_experiments.ipynb
│   └── inference_demo.ipynb
├── reports/
│   ├── case_study_report.pdf
│   └── presentation_slides.pptx
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Evaluation Metrics
- **SSIM**: Structural Similarity Index  
- **LPIPS**: Learned Perceptual Image Patch Similarity  
- **FID**: Fréchet Inception Distance  

---

## Contributing
Contributions are welcome! Open an issue or submit a pull request for improvements.

---

## License
This project is licensed under the MIT License.
