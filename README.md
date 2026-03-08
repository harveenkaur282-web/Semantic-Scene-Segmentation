# OffRoad Scene Segmentation — Duality Hackathon

> **Semantic segmentation of off-road terrain** using a fine-tuned SegFormer-B2 transformer with two-stage training, Test-Time Augmentation, and CRF-based post-processing.

---
# DRIVE LINK THAT HAS PITCH VIDEO, DEMO VIDEO, ANSWERS OF ASSESSMENT, AND THE MAIN NOTEBOOK FILE = https://drive.google.com/drive/folders/16_LIy5N4i8Z7vOtLKY8HVT3izpA6StNK

## Idea Title

**OffRoad Terrain Segmentation with SegFormer-B2 & Two-Stage Fine-Tuning**

---

## Idea Description

Off-road autonomous navigation requires robust understanding of unstructured terrain — distinguishing dirt trails from rocks, grass, sky, and obstacles. This project tackles the **Duality Hackathon's Off-Road Segmentation Challenge**, producing pixel-level semantic maps across 6 terrain classes from RGB drone/camera images.

The approach uses a **two-stage training strategy**:
- **Stage 1** — Backbone frozen, only the decode head is trained to stabilize early learning.
- **Stage 2** — Full fine-tuning with a lower learning rate, combined loss (CE + Dice), label smoothing, mixed-precision training, and gradient clipping.

At inference, **Test-Time Augmentation (TTA)** with horizontal flipping and multi-scale averaging is applied, followed by **CRF smoothing** to sharpen spatial boundaries. The final model achieves **mIoU of 0.5155** across 5 present test classes.

---

## Technical Details

### Technologies Used

| Category | Stack |
|---|---|
| **Core Framework** | Python 3, PyTorch |
| **Transformer Model** | HuggingFace Transformers — `SegformerForSemanticSegmentation` |
| **Pretrained Backbone** | `nvidia/segformer-b2-finetuned-ade-512-512` |
| **Image Augmentation** | Albumentations v2.0+ |
| **Metrics** | torchmetrics — `MulticlassJaccardIndex` (mIoU) |
| **Post-Processing** | pydensecrf (CRF smoothing) |
| **Visualization** | matplotlib, seaborn, OpenCV |
| **Training Utilities** | tqdm, scikit-learn |
| **Environment** | Kaggle Notebooks (T4 GPU) |

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Input Resolution | 512 × 512 |
| Batch Size | 8 |
| Optimizer | AdamW (lr=6e-5, wd=1e-2) |
| Scheduler | CosineAnnealingLR (T_max=20, eta_min=1e-6) |
| Epochs | 20 (Stage 1) + 20 (Stage 2) |
| Loss | CE (label_smoothing=0.1) + Dice (50/50) |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | torch.cuda.amp (FP16) |

---

## Architecture Overview

```
Input RGB Image (512×512)
        │
        ▼
┌──────────────────────────────┐
│   SegFormer-B2 Backbone      │  ← Hierarchical Transformer Encoder
│   (Mix Transformer Encoder)  │     4 stages, overlapping patch merging
└──────────────┬───────────────┘
               │  Multi-scale feature maps
               ▼
┌──────────────────────────────┐
│   All-MLP Decode Head        │  ← Lightweight MLP aggregator
│   (num_labels=6)             │     Linear projection + upsampling
└──────────────┬───────────────┘
               │  Low-res logits (128×128)
               ▼
     F.interpolate → 512×512
               │
               ▼
┌──────────────────────────────┐
│   CombinedLoss               │  ← CrossEntropy (label_smooth=0.1)
│   (Training only)            │     + Soft Dice Loss
└──────────────────────────────┘

 ── Inference Pipeline ──────────────────────────────────────────
  Original Image + H-Flip  →  TTA average logits
        │
        ▼
  argmax(probs)  →  CRF Smoothing  →  Final Segmentation Mask
```

### Six Terrain Classes

| Class ID | Label | Description |
|---|---|---|
| 0 | Background | Undefined / unlabeled |
| 1 | Dirt/Trail | Traversable off-road paths |
| 2 | Grass/Veg | Vegetation (absent in test set) |
| 3 | Rock | Rocky terrain |
| 4 | Sky | Open sky regions |
| 5 | Obstacle | Non-traversable obstacles |

---

## Database / Data Used

- **No external database.** Dataset provided by Duality Hackathon organizers (Google Cloud Storage).
- Training set: RGB images + RGB-encoded semantic masks (`Color_Images/` + `Segmentation/`)
- Mask encoding: Original label values `{0, 1, 2, 3, 27, 39}` → remapped to `{0, 1, 2, 3, 4, 5}`
- Split: Train / Val / Test directories provided pre-split.

---

## Third-Party Integrations

| Integration | Purpose |
|---|---|
| **HuggingFace Hub** | Downloads pretrained `nvidia/segformer-b2-finetuned-ade-512-512` weights |
| **Kaggle Notebooks** | Training environment with T4 GPU access |
| **Google Cloud Storage** | Dataset hosting (provided by hackathon organizers) |

> No API keys or external services are required to run inference after downloading model weights.

---

##  Results

| Metric | Score |
|---|---|
| mIoU (5 existing classes) | **0.5155** |
| Dirt/Trail IoU | 0.4961 |
| Sky IoU | 0.9819 |
| Obstacle IoU | 0.6165 |
| Background IoU | 0.4249 |
| Rock IoU | 0.0582 |
| Inference Speed | ~X ms/image on T4 |

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install transformers timm
pip install segmentation-models-pytorch
pip install albumentations
pip install torchmetrics
pip install pydensecrf
pip install opencv-python matplotlib seaborn tqdm scikit-learn
```

### Dataset Setup

Download the two zip files directly from the Duality Hackathon Google Cloud Storage bucket:

| File | Link | Contents |
|---|---|---|
| **Training + Validation Set** | [Offroad_Segmentation_Training_Dataset.zip](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip) | RGB images + segmentation masks for train & val splits |
| **Test Set** | [Offroad_Segmentation_testImages.zip](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_testImages.zip) | RGB images + masks for evaluation |

Or download via terminal:

```bash
mkdir -p data

# Training set
wget -O train.zip "https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip"

# Test set
wget -O test.zip "https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_testImages.zip"

# Extract both
unzip train.zip -d data/
unzip test.zip -d data/
```

After extraction, your `data/` folder should look like:

```
data/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
└── Offroad_Segmentation_testImages/
    ├── Color_Images/
    └── Segmentation/
```

### Training

```bash
# Stage 1 — Frozen backbone
python src/train.py --stage 1 --epochs 20

# Stage 2 — Full fine-tuning
python src/train.py --stage 2 --epochs 20 --checkpoint checkpoints/stage1_best.pth
```

### Inference

```bash
python src/predict.py \
  --checkpoint checkpoints/best_model.pth \
  --input data/Offroad_Segmentation_testImages/Color_Images/ \
  --output outputs/predictions/
```

---

## 📁 Repo Structure

See [`REPO_STRUCTURE.md`](./REPO_STRUCTURE.md) for the full breakdown.

---

## 📄 License

This project was developed for the **Duality Hackathon**. Dataset copyright belongs to Duality AI.
