# Semantic Scene Segmentation for Off-Road Terrain

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  Pixel-level semantic segmentation of off-road terrain using a <strong>DINOv2 + UNet++ pipeline</strong>, trained on a 10-class outdoor scene dataset.
</p>

---

## Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [License](#-license)

---

## Overview

This project tackles **semantic scene segmentation** in challenging off-road environments — forests, rocky terrain, dry grasslands, and dense vegetation. The goal is to accurately classify every pixel in an image into one of 10 terrain categories.

The pipeline evolved through two stages:

| Stage | Architecture | mIoU |
|---|---|---|
| **Baseline** | DINOv2 ViT-S/14 + Custom ConvNeXt Head | 0.3169 |
| **Improved** | UNet++ (ResNet-50 encoder, ImageNet weights) | 0.5452+ |

---

## Dataset

The dataset is sourced from the **Duality Hackathon Off-Road Segmentation Dataset** and organized as follows:

```
data/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/        # 2,857 RGB images (.png)
│   │   └── Segmentation/        # 2,857 masks (.png)
│   └── val/
│       ├── Color_Images/        # 317 RGB images (.png)
│       └── Segmentation/        # 317 masks (.png)
└── Offroad_Segmentation_testImages/
    └── Color_Images/            # 1,002 test images (no masks)
```

### Dataset Split Summary

| Split | Images | Masks |
|---|---|---|
| Train | 2,857 | 2,857 |
| Validation | 317 | 317 |
| Test | 1,002 | None |

### Class Mapping

Masks use raw integer values that are remapped to sequential class IDs (0–9):

| Class ID | Class Name | Raw Mask Value |
|---|---|---|
| 0 | Background | 0 |
| 1 | Trees | 100 |
| 2 | Lush Bushes | 200 |
| 3 | Dry Grass | 300 |
| 4 | Dry Bushes | 500 |
| 5 | Ground Clutter | 550 |
| 6 | Logs | 700 |
| 7 | Rocks | 800 |
| 8 | Landscape | 7100 |
| 9 | Sky | 10000 |

---

## Model Architecture

### Baseline — DINOv2 + ConvNeXt Segmentation Head

- **Backbone:** Facebook's DINOv2 ViT-Small/14, pretrained via self-supervised learning, **frozen** during training
- **Head:** Custom lightweight `SegmentationHeadConvNeXt` — reshapes patch tokens into a spatial feature map, applies depthwise 7×7 convolutions with GELU activations, and outputs per-pixel class logits
- **Why DINOv2?** Its self-supervised Vision Transformer features generalize extremely well to dense prediction tasks without requiring full fine-tuning

### Improved — UNet++ with ResNet-50 Encoder

- **Architecture:** `UNet++` from `segmentation-models-pytorch`
- **Encoder:** ResNet-50, pretrained on ImageNet — provides rich hierarchical features
- **Why UNet++?** Nested skip connections capture fine-grained spatial detail and multi-scale context, making it highly effective for complex terrain segmentation

---

## Training Configuration

### Baseline

| Parameter | Value |
|---|---|
| Batch Size | 2 |
| Image Size | 266 × 476 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Backbone | DINOv2 ViT-S/14 (frozen) |
| Augmentation | None |
| Device | GPU (CUDA) |

### Improved

| Parameter | Value |
|---|---|
| Batch Size | 4 |
| Image Size | 448 × 768 |
| Learning Rate | 1e-4 (max 1e-3 with scheduler) |
| Optimizer | AdamW (weight decay 1e-2) |
| Loss Function | 0.5 × Dice Loss + 0.5 × Focal Loss |
| Scheduler | OneCycleLR |
| Epochs | 20+ |
| Mixed Precision | AMP (autocast + GradScaler) |
| Device | GPU (CUDA) |

### Data Augmentation (Improved Stage)

Applied using **Albumentations** (synchronized image + mask transforms):

| Technique | Parameters |
|---|---|
| Horizontal Flip | p=0.5 |
| Random Brightness & Contrast | p=0.2 |
| Color Jitter | p=0.2 |
| Shift, Scale, Rotate | shift=0.1, scale=0.1, rotate=±15°, p=0.5 |
| Normalization | ImageNet mean/std |

---

## Results

### Baseline Training History

| Epoch | Train Loss | Val Loss | Val mIoU |
|---|---|---|---|
| 1 | 1.1796 | 0.9950 | 0.2377 |
| 2 | 0.9495 | 0.9140 | 0.2626 |
| 3 | 0.8975 | 0.8790 | 0.2798 |
| 4 | 0.8714 | 0.8582 | 0.2947 |
| 5 | 0.8551 | 0.8449 | 0.3025 |
| 6 | 0.8436 | 0.8358 | 0.3044 |
| 7 | 0.8351 | 0.8279 | 0.3087 |
| 8 | 0.8285 | 0.8223 | 0.3088 |
| 9 | 0.8231 | 0.8178 | 0.3122 |
| **10** | **0.8187** | **0.8132** | **0.3169** |

### Per-Class IoU (Baseline)

| Class | Name | mIoU |
|---|---|---|
| 0 | Background | 0.2629 |
| 1 | Trees | 0.3750 |
| 2 | Lush Bushes | 0.3545 |
| 3 | Dry Grass | **0.4409** |
| 4 | Dry Bushes | 0.0825 |
| 5 | Ground Clutter | ~0.065 |
| **—** | **Overall mIoU** | **0.3169** |

> Dry Grass achieved the highest per-class IoU. Dry Bushes and Ground Clutter were the most challenging — likely due to visual similarity with neighbouring terrain and class imbalance.

### Improved Model — Validation mIoU Progress

| Epoch | Val mIoU |
|---|---|
| 1 | 0.5016 |
| 2 | 0.5085 |
| 3 | 0.5305 |
| **6** | **0.5452** Best |

---

## Project Structure

```
Semantic-Scene-Segmentation/
│
├── models/                        # Saved model checkpoints
├── cofig.py                       # Configuration: paths, hyperparameters, class map
├── dataset.py                     # Custom PyTorch Dataset with mask conversion
├── train.py                       # Main training entry point
├── training_segmentation.py       # Training loop, loss, optimizer, scheduler
├── testing_segmentation.py        # Inference and prediction export
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/harveenkaur282-web/Semantic-Scene-Segmentation.git
cd Semantic-Scene-Segmentation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download and extract the Duality Hackathon Off-Road Segmentation datasets into the `data/` directory:

- [Training Dataset (~2.6 GB)](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip)
- [Test Dataset (~1.0 GB)](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_testImages.zip)

Expected structure after extraction:
```
data/
├── Offroad_Segmentation_Training_Dataset/
└── Offroad_Segmentation_testImages/
```

### 4. Configure Paths

Edit `cofig.py` to set your local dataset paths and hyperparameters.

### 5. Train

```bash
python train.py
```

### 6. Run Inference

```bash
python testing_segmentation.py
```

---

## Requirements

```
torch
torchvision
segmentation-models-pytorch
albumentations
numpy
Pillow
opencv-python
tqdm
matplotlib
```

Install all at once:

```bash
pip install -r requirements.txt
```

A **CUDA-capable GPU** is strongly recommended (training was performed on a single GPU with mixed precision).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
