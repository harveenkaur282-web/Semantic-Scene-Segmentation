# Semantic Scene Segmentation

## Table of Contents

- [Overview](#overview)
- [Environment & Dependency Requirements](#environment--dependency-requirements)
- [Setup Instructions](#setup-instructions)
- [How to Run Training](#how-to-run-training)
- [How to Run Testing / Inference](#how-to-run-testing--inference)
- [Reproducing Final Results](#reproducing-final-results)
- [Expected Outputs & How to Interpret Them](#expected-outputs--how-to-interpret-them)

---

## Overview

This project implements a semantic scene segmentation model that assigns a class label to every pixel in an image. Images are resized to **256×256** pixels before being fed through the pipeline. The repository includes a full training loop, a segmentation training script, and a dedicated testing/evaluation script.

---

## Environment & Dependency Requirements

### Python Version
- Python 3.8 or higher (3.9+ recommended)

### Core Dependencies

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
Pillow>=9.0.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```

Install all dependencies via:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision numpy Pillow opencv-python matplotlib scikit-learn tqdm
```

### GPU Support (Recommended)

For faster training, a CUDA-compatible GPU is strongly recommended. Verify your CUDA setup with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/harveenkaur282-web/Semantic-Scene-Segmentation.git
cd Semantic-Scene-Segmentation
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

##3 4. Dataset

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

###  Download the Dataset

Download and extract the Duality Hackathon Off-Road Segmentation datasets into the `data/` directory:

- [Training Dataset (~2.6 GB)](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip)
- [Test Dataset (~1.0 GB)](https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_testImages.zip)
---

## How to Run Training

The main training entry point is `train.py`, which implements the full forward pass pipeline with image resizing to 256×256.

```bash
python train.py
```

### Common Training Arguments (if supported)

```bash
python train.py \
  --data_dir ./data \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.001 \
  --save_dir ./checkpoints
```

Alternatively, if the project uses `training_segmentation.py` as the script provided by the course/instructor:

```bash
python training_segmentation.py
```

> **Note:** Check the top of each script for configurable parameters (paths, hyperparameters, number of classes, etc.) and adjust them to match your dataset before running.

---

## How to Run Testing / Inference

Once a model has been trained and a checkpoint saved, run evaluation using `testing_segmentation.py`:

```bash
python testing_segmentation.py
```

### Common Testing Arguments (if supported)

```bash
python testing_segmentation.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data_dir ./data/test \
  --output_dir ./predictions
```

This will load the trained model weights, run inference on the test set, and save the predicted segmentation masks.

---

## Reproducing Final Results

To reproduce the final reported results from scratch:

1. **Clone and set up** the environment as described above.
2. **Prepare the dataset** in the expected folder structure.
3. **Run training** using the provided script:
   ```bash
   python train.py
   ```
4. **Identify the best checkpoint** (typically saved as `best_model.pth` or the final epoch checkpoint in `./checkpoints/`).
5. **Run evaluation** on the test set:
   ```bash
   python testing_segmentation.py --checkpoint ./checkpoints/best_model.pth
   ```
6. Compare the output metrics (mIoU, pixel accuracy) against the reported values.

> **Tip:** Ensure that image preprocessing (resize to 256×256, normalization values) in the testing script matches exactly what was used during training.

---

## Expected Outputs & How to Interpret Them

### During Training

You should see per-epoch output similar to:

```
Epoch [1/50] | Loss: 1.2345 | Train Acc: 0.72 | Val mIoU: 0.45
Epoch [2/50] | Loss: 1.0821 | Train Acc: 0.76 | Val mIoU: 0.49
...
```

- **Loss**: Should decrease steadily. If it plateaus early, consider adjusting the learning rate.
- **Train Accuracy**: Pixel-level accuracy on training data; expected to rise over epochs.
- **Val mIoU**: Mean Intersection over Union on the validation set — the primary metric for segmentation quality. Higher is better (max = 1.0).

### During Testing / Inference

- **Predicted segmentation masks** are saved as image files (typically `.png`) where each pixel value corresponds to a class label (e.g., 0 = background, 1 = road, 2 = sky, etc.).
- **Color-coded visualizations** may be generated if the script includes a colormap, making it easy to visually inspect which regions were assigned to which class.
- **Evaluation metrics** printed to console typically include:
  - `mIoU` (Mean Intersection over Union) — overall segmentation performance across all classes
  - `Pixel Accuracy` — percentage of correctly labeled pixels
  - Per-class IoU — breakdown of accuracy per semantic category

### Interpreting Segmentation Masks

Each pixel in the output mask is assigned an integer class ID. For example:

| Pixel Value | Class       |
|-------------|-------------|
| 0           | Background  |
| 1           | Road        |
| 2           | Sky         |
| 3           | Person      |
| ...         | ...         |

Refer to the dataset's class definitions (or the label mapping defined in the code) to identify what each class ID represents.

---
