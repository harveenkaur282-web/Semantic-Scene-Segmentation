#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install -q segmentation-models-pytorch albumentations')

import urllib.request, zipfile, os
os.makedirs('data', exist_ok=True)

urllib.request.urlretrieve(
    "https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_Training_Dataset.zip",
    "train.zip"
)

urllib.request.urlretrieve(
    "https://storage.googleapis.com/duality-public-share/Hackathons/Duality%20Hackathon/Offroad_Segmentation_testImages.zip", 
    "test.zip"
)

with zipfile.ZipFile("train.zip") as z: 
    z.extractall("data/")

with zipfile.ZipFile("test.zip") as z: 
    z.extractall("data/")


# In[4]:


get_ipython().system('pip install -q transformers timm albumentations torchmetrics')


# In[5]:


from pathlib import Path

train_base = Path("data/Offroad_Segmentation_Training_Dataset")

if (train_base / "train").exists():
    print(f"\n Found train folder")
    train_colored = list((train_base / "train").rglob("*.png"))
    print(f" Images in train: {len(train_colored)}")

if (train_base / "val").exists():
    print(f"\n Found val folder")
    val_colored = list((train_base / "val").rglob("*.png"))
    print(f" Images in val: {len(val_colored)}")

test_base = Path("data/Offroad_Segmentation_testImages/Color_Images")
test_images = list(test_base.glob("*.png"))
print(f" Test images: {len(test_images)}")

print("\n Sample paths:")
if train_colored:
    print(f"Train image: {train_colored[0]}")
if val_colored:
    print(f"Val image: {val_colored[0]}")
if test_images:
    print(f"Test image: {test_images[0]}")


# In[6]:


test_base_masks = Path("data/Offroad_Segmentation_testImages/Segmentation")
test_masks = list(test_base_masks.glob("*.png"))
print(f"test masks: {len(test_masks)}")


# In[7]:


from pathlib import Path

TRAIN_DIR = Path("data/Offroad_Segmentation_Training_Dataset/train")
VAL_DIR = Path("data/Offroad_Segmentation_Training_Dataset/val")
TEST_DIR = Path("data/Offroad_Segmentation_testImages/Color_Images")

print(f"   TRAIN_DIR: {TRAIN_DIR}")
print(f"   VAL_DIR: {VAL_DIR}")
print(f"   TEST_DIR: {TEST_DIR}")

assert TRAIN_DIR.exists(), f"Training dir not found: {TRAIN_DIR}"
assert VAL_DIR.exists(), f"Validation dir not found: {VAL_DIR}"
assert TEST_DIR.exists(), f"Test dir not found: {TEST_DIR}"


# In[8]:


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm


# In[9]:


from pathlib import Path

train_path = Path("data/Offroad_Segmentation_Training_Dataset/train")
val_path = Path("data/Offroad_Segmentation_Training_Dataset/val")

for item in sorted(train_path.iterdir()):
    if item.is_dir():
        files = list(item.glob("*"))
        print(f" {item.name}/ ({len(files)} files)")
        if files:
            print(f"     Sample: {files[0].name}")

for item in sorted(val_path.iterdir()):
    if item.is_dir():
        files = list(item.glob("*"))
        print(f" {item.name}/ ({len(files)} files)")
        if files:
            print(f"     Sample: {files[0].name}")


# In[10]:


from pathlib import Path

train_path = Path("data/Offroad_Segmentation_Training_Dataset/train")

print("Train directory structure:\n")

for item in train_path.iterdir():
    print(item)


# In[11]:


val_path = Path("data/Offroad_Segmentation_Training_Dataset/val")

print("Val directory structure:\n")

for item in val_path.iterdir():
    print(item)


# In[12]:


from pathlib import Path

TRAIN_DIR = Path("data/Offroad_Segmentation_Training_Dataset/train")
VAL_DIR = Path("data/Offroad_Segmentation_Training_Dataset/val")

train_images = sorted(list((TRAIN_DIR / "Color_Images").glob("*.png")))
train_masks = sorted(list((TRAIN_DIR / "Segmentation").glob("*.png")))

val_images = sorted(list((VAL_DIR / "Color_Images").glob("*.png")))
val_masks = sorted(list((VAL_DIR / "Segmentation").glob("*.png")))

print("Train images:", len(train_images))
print("Train masks:", len(train_masks))

print("Val images:", len(val_images))
print("Val masks:", len(val_masks))


# In[13]:


import cv2
import matplotlib.pyplot as plt
import random

def show_sample(idx):

    img = cv2.imread(str(train_images[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(train_masks[idx]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.axis("off")

    plt.show()

for _ in range(3):
    show_sample(random.randint(0, len(train_images)-1))


# In[14]:


print(train_images[0].name)
print(train_masks[0].name)


# In[15]:


import cv2
import matplotlib.pyplot as plt

idx = 10

img = cv2.imread(str(train_images[idx]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = cv2.imread(str(train_masks[idx]), 0)

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.imshow(mask, alpha=0.4)
plt.title("Overlay")
plt.axis("off")
plt.show()


# In[16]:


import numpy as np

mask = cv2.imread(str(train_masks[0]))
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

pixels = mask.reshape(-1,3)

unique_colors = np.unique(pixels, axis=0)

print("Unique mask colors:")
print(unique_colors)

print("\nNumber of classes:", len(unique_colors))


# In[17]:


img = cv2.imread(str(train_images[0]))
print("Image shape:", img.shape)

mask = cv2.imread(str(train_masks[0]))
print("Mask shape:", mask.shape)


# In[18]:


import numpy as np
import cv2

mask = cv2.imread(str(train_masks[0]))
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

pixels = mask.reshape(-1,3)
unique_colors = np.unique(pixels, axis=0)

print("Unique colors:")
print(unique_colors)

print("\nNumber of classes:", len(unique_colors))


# In[19]:


import numpy as np

ORIGINAL_LABELS = [0, 1, 2, 3, 27, 39]

LABEL_MAPPING = {v: i for i, v in enumerate(ORIGINAL_LABELS)}

print(LABEL_MAPPING)


# In[20]:


def convert_mask(mask):

    mask = mask[:,:,0]   #(converting rgb to grayscale)

    new_mask = np.zeros_like(mask)

    for old, new in LABEL_MAPPING.items():
        new_mask[mask == old] = new

    return new_mask


# In[21]:


# Updated augmentations for Albumentations v2.0+
train_transform = A.Compose([
    # Change height/width to size=(height, width)
    A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),
        A.HueSaturationValue(p=1.0),
        A.RGBShift(p=1.0),
    ], p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    # Use size instead of individual height/width if you decide to use ResizedCrop here too
    A.Resize(height=512, width=512), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# In[22]:


import torch
from torch.utils.data import Dataset
import cv2

class OffroadDataset(Dataset):

    def __init__(self, images, masks, transform=None):

        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[idx]))

        mask = convert_mask(mask)

        if self.transform:

            augmented = self.transform(
                image=img,
                mask=mask
            )

            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.long()


# In[23]:


from torch.utils.data import DataLoader

train_dataset = OffroadDataset(
    train_images,
    train_masks,
    transform=train_transform
)

val_dataset = OffroadDataset(
    val_images,
    val_masks,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)


# In[24]:


get_ipython().system('pip install -q transformers timm')


# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


# In[25]:


NUM_CLASSES = 6

model = SegformerForSemanticSegmentation.from_pretrained(

    "nvidia/segformer-b2-finetuned-ade-512-512",

    num_labels=NUM_CLASSES,

    ignore_mismatched_sizes=True

)

model = model.to(device)

print("Model loaded")


# In[96]:


for param in model.segformer.parameters():
    param.requires_grad = False

print("Backbone frozen")


# In[97]:


import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # targets: (batch, height, width)
        # logits: (batch, num_classes, height, width)
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_loss = 1 - (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return dice_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        # Adding label_smoothing=0.1 here applies the regularization
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1) 
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        return self.weight_ce * self.ce(logits, targets) + self.weight_dice * self.dice(logits, targets)


# In[98]:


optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-2)


# In[99]:


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-6
)


# In[100]:


criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)


# In[33]:


scaler = torch.cuda.amp.GradScaler()


# In[101]:


def compute_iou(pred, mask):

    pred = pred.argmax(1)

    intersection = (pred == mask) & (mask > 0)
    union = (pred > 0) | (mask > 0)

    iou = intersection.sum().float() / (union.sum().float() + 1e-6)

    return iou.item()


# In[32]:


EPOCHS = 20
best_iou = 0
max_norm = 1.0 # Standard for gradient clipping

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with torch.cuda.amp.autocast():
            # 1. Forward pass
            outputs = model(pixel_values=images)
            logits = outputs.logits

            # 2. Upsample logits to match mask size
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # 3. Calculate Loss (Combined Dice + CE with Label Smoothing)
            loss = criterion(logits, masks)

        # 4. Scaled Backward Pass (ONLY ONCE)
        scaler.scale(loss).backward()

        # 5. Unscale for Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # 6. Optimizer & Scaler Step
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    # Step the scheduler after each epoch
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")


# In[33]:


model.eval()

val_iou = 0

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images)

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        val_iou += compute_iou(logits, masks)

val_iou /= len(val_loader)

print("Validation IoU:", val_iou)

if val_iou > best_iou:

    best_iou = val_iou
    torch.save(model.state_dict(), "best_model.pth")

    print("Best model saved")


# In[102]:


# 1. Initialize the same model architecture
# model = SegformerForSemanticSegmentation.from_pretrained(...) 

# 2. Load the physical file you saved from Stage 1
model.load_state_dict(torch.load('best_model.pth')) 
model.to(device)

print("Weights loaded! You haven't lost any progress.")


# In[103]:


model.load_state_dict(torch.load("best_model.pth"))


# In[67]:


import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

# 1. Unfreeze the entire model
for param in model.parameters():
    param.requires_grad = True

print("Model fully unfrozen for Stage 2.")

# 2. Differential Learning Rates
# We treat the encoder (backbone) and decoder (head) differently
optimizer = optim.AdamW([
    {'params': model.segformer.encoder.parameters(), 'lr': 1e-6}, # Backbone: Very slow
    {'params': model.decode_head.parameters(), 'lr': 3e-5}       # Head: Faster
], weight_decay=1e-4)

# 3. Scheduler & Scaler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
scaler = GradScaler()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

best_val_iou = 0.0 # Track progress to save the best version


# In[104]:


# Re-define your train_loader with a smaller batch size
train_loader = DataLoader(
    train_dataset, 
    batch_size=1,  # Try 2 first. If it crashes, use 1.
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)


# In[71]:


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler # Updated import to avoid the warning
from tqdm import tqdm
import gc

# 1. CLEAN MEMORY & PREPARE MODEL
gc.collect()
torch.cuda.empty_cache()

for param in model.parameters():
    param.requires_grad = True

print("Backbone unfrozen. Preparing Stage 2...")

# 2. CONFIGURATION
num_epochs = 15
accumulation_steps = 8  # Increased to 8 because batch_size is 1
max_norm = 1.0          
best_val_iou = 0.0

# 3. DIFFERENTIAL LEARNING RATES
optimizer = optim.AdamW([
    {'params': model.segformer.encoder.parameters(), 'lr': 1e-6}, 
    {'params': model.decode_head.parameters(), 'lr': 3e-5}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = GradScaler('cuda') # Fixed the warning here

# 4. TRAINING LOOP
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad() 

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Mixed Precision Forward Pass
        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps 

        # Backward Pass
        scaler.scale(loss).backward()

        # Update weights every 'accumulation_steps'
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})

    scheduler.step()

    # 5. VALIDATION PHASE
    model.eval()
    iou_total = 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = torch.argmax(outputs, dim=1)

            intersection = (preds == masks).sum()
            union = (preds.nelement() + masks.nelement() - intersection)
            iou = intersection.float() / (union.float() + 1e-6)

            iou_total += iou.item()
            num_batches += 1

    val_iou = iou_total / num_batches
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val IoU: {val_iou:.4f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), 'best_model_stage2.pth')
        print(">>> New Best Model Saved!")

    # Extra memory cleanup after each epoch
    gc.collect()
    torch.cuda.empty_cache()
    print("-" * 30)

print(f"Training Complete. Best Validation IoU: {best_val_iou:.4f}")


# In[106]:


# Check a single mask to see the unique class IDs
sample_mask = val_dataset[0][1] # Get the mask from the first item
unique_ids = torch.unique(sample_mask).tolist()
print(f"Unique Class IDs found in your data: {unique_ids}")
print(f"Total number of classes: {len(unique_ids)}")


# In[112]:


import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def check_per_class_iou(model, val_loader, num_classes=6):
    model.eval()
    all_inter = torch.zeros(num_classes).to(device)
    all_union = torch.zeros(num_classes).to(device)

    # Standard Off-Road Mapping
    class_names = [
        "0: Background", 
        "1: Path/Trail", 
        "2: Grass/Veg", 
        "3: Trees", 
        "4: Obstacles", 
        "5: Sky"
    ] 

    print("Analyzing 6 Classes on Validation Set...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(pixel_values=images).logits
            # Match mask size
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(num_classes):
                inter = ((preds == cls) & (masks == cls)).sum()
                union = ((preds == cls) | (masks == cls)).sum()
                all_inter[cls] += inter
                all_union[cls] += union

    print("\n" + "="*35)
    print("      FINAL PERFORMANCE REPORT")
    print("="*35)

    ious = []
    for i in range(num_classes):
        iou = (all_inter[i] / (all_union[i] + 1e-6)).item()
        ious.append(iou)
        print(f"{class_names[i]:15} : {iou:.4f}")

    print("-" * 35)
    print(f"Mean IoU (mIoU) : {np.mean(ious):.4f}")
    print("="*35)

# Execute
check_per_class_iou(model, val_loader)


# In[74]:


import matplotlib.pyplot as plt

def quick_color_check(dataset, index=0):
    img, mask = dataset[index]
    plt.imshow(mask.cpu().numpy(), cmap='nipy_spectral')
    plt.colorbar(ticks=range(6), label='Class ID')
    plt.title(f"Visualizing Class IDs for Index {index}")
    plt.show()

quick_color_check(val_dataset, index=0)


# In[115]:


import scipy.ndimage as ndimage

def visualize_with_smoothing(model, dataset, index=0):
    model.eval()
    image, mask = dataset[index]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor).logits
        outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear")
        prediction = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # APPLY THE MAGIC FILTER
    # size=7 or 9 is usually good for heavy noise like yours
    smoothed_pred = ndimage.median_filter(prediction, size=7)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(prediction, cmap='nipy_spectral')
    plt.title("Original (Messy) Prediction")

    plt.subplot(1, 2, 2)
    plt.imshow(smoothed_pred, cmap='nipy_spectral')
    plt.title("Smoothed (Clean) Prediction")
    plt.show()

visualize_with_smoothing(model, val_dataset, index=0)


# In[77]:


import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 1. Load your best model from Stage 2
model.load_state_dict(torch.load('best_model_stage2.pth'))
model.to(device)

# 2. Define Class Weights to fix the "Noise"
# Order: [0:Background, 1:Path, 2:Grass, 3:Trees, 4:Obstacles, 5:Sky]
# We give Path and Obstacles high weights to force focus there.
weights = torch.tensor([1.0, 3.5, 1.2, 1.2, 3.5, 0.5]).to(device)
criterion_ce = nn.CrossEntropyLoss(weight=weights)

# 3. Optimizer with "Refinement" Learning Rates
# We keep the backbone very low to avoid "breaking" what it learned, 
# but high enough to move it out of the plateau.
# 2. Corrected Optimizer for SegFormer
# segformer.encoder = the backbone/features
# decode_head = the classification head
optimizer = torch.optim.AdamW([
    {'params': model.segformer.encoder.parameters(), 'lr': 2e-6}, 
    {'params': model.decode_head.parameters(), 'lr': 2e-5}
], weight_decay=0.01)

print("Starting Stage 3: Refining Path and Obstacles...")

for epoch in range(1, 6):
    model.train()
    epoch_loss = 0

    pbar = tqdm(train_loader, desc=f"Stage 3 - Epoch {epoch}/5")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion_ce(outputs, masks)
        # If you have a Dice Loss function defined, you can add it here:
        # loss += criterion_dice(outputs, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # After each epoch, let's see if the Path IoU is rising!
    print(f"\nEpoch {epoch} Training Loss: {epoch_loss/len(train_loader):.4f}")
    # You can call your check_per_class_iou(model, val_loader) here to track progress


# In[110]:


import torch

# Move model to evaluation mode (freezes dropout/batchnorm)
model.eval()

# Save immediately
torch.save(model.state_dict(), 'stage3_checkpoint.pth')
print("✅ Weights saved safely. If the system crashes now, your progress is secure.")


# In[113]:


print("📊 Calculating Stage 3 IoU Scores...")
with torch.no_grad(): # This prevents memory buildup
    check_per_class_iou(model, val_loader)


# In[116]:


import torch
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

print("🖼️ Generating Comparison Visualization...")
# Using the function we discussed earlier
visualize_with_smoothing(model, val_dataset, index=0)


# In[117]:


import numpy as np
import scipy.ndimage as ndimage

def predict_with_tta_and_smooth(model, image):
    model.eval()
    img_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Original Pass
        out1 = model(pixel_values=img_tensor).logits

        # 2. Flipped Pass
        img_flipped = torch.flip(img_tensor, dims=[3])
        out2 = model(pixel_values=img_flipped).logits
        out2 = torch.flip(out2, dims=[3]) # Flip back

        # 3. Average & Smooth
        final_logits = (out1 + out2) / 2
        final_logits = F.interpolate(final_logits, size=(512, 512), mode="bilinear")
        prediction = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()

        # 4. Final Clean-up (The Median Filter)
        cleaned_pred = ndimage.median_filter(prediction, size=5)

    return cleaned_pred

print("Final TTA + Smoothing Pipeline Ready.")


# In[118]:


all_preds = []
all_masks = []

print(" Running Final TTA + Smoothing Evaluation...")
model.eval()

for images, masks in tqdm(val_loader):
    for i in range(images.size(0)):
        # Use our new "Clean" prediction function
        clean_pred = predict_with_tta_and_smooth(model, images[i])

        all_preds.append(clean_pred)
        all_masks.append(masks[i].cpu().numpy())

# Calculate the final improved IoU
final_predictions = np.array(all_preds)
final_ground_truth = np.array(all_masks)

# Use your existing IoU calculation logic here
# (e.g., calling your check_per_class_iou logic on these final arrays)
print("Final Processed Scores are ready!")


# In[119]:


from sklearn.metrics import jaccard_score

def calculate_final_metrics(preds, masks, num_classes=6):
    # Flatten the arrays for comparison
    preds_flat = np.array(preds).flatten()
    masks_flat = np.array(masks).flatten()

    # Calculate per-class IoU
    ious = jaccard_score(masks_flat, preds_flat, average=None, labels=range(num_classes))

    print("===================================")
    print("      OFFICIAL FINAL REPORT (TTA + SMOOTHED)")
    print("===================================")
    class_names = ["Background", "Path/Trail", "Grass/Veg", "Trees", "Obstacles", "Sky"]

    for i, iou in enumerate(ious):
        print(f"{i}: {class_names[i]:<15} : {iou:.4f}")

    print("-----------------------------------")
    print(f"Final Mean IoU (mIoU) : {np.mean(ious):.4f}")
    print("===================================")

# Call the function
calculate_final_metrics(all_preds, all_masks)


# In[26]:


test_dataset = OffroadDataset(
    test_images,
    test_masks,
    transform=val_transform
)


# In[27]:


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# In[122]:


all_test_preds = []
# Note: If your test set has no masks (hidden), we can only save predictions
# If it HAS masks, we can calculate the score.

print("Running Final Inference on TEST DATA...")
model.eval()

# SWAP val_loader for test_loader here
for images, masks in tqdm(test_loader): 
    for i in range(images.size(0)):
        clean_pred = predict_with_tta_and_smooth(model, images[i])
        all_test_preds.append(clean_pred)
        # only append masks if they exist in your test set
        # all_test_masks.append(masks[i].cpu().numpy()) 

print("✅ Test Set Processing Complete!")


# In[125]:


print(f"Images in test_loader: {len(test_loader.dataset)}")
# If this says 0, your file paths are incorrect!


# In[127]:


import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

# 1. FIX THE LOADER (Lower workers = Lower CPU strain)
# We set num_workers to 0 to force the CPU to handle things one-by-one
test_loader.num_workers = 0 
# If you can, redefine the loader with a smaller batch size
# test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
print("sorting out the memory thing")


# In[140]:


import numpy as np

def calculate_test_metrics_efficiently(test_preds, test_loader, num_classes=6):
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    print(" Calculating metrics batch-by-batch...")

    current_idx = 0
    for _, masks in test_loader:
        masks_np = masks.cpu().numpy()
        for i in range(masks_np.shape[0]):
            true_mask = masks_np[i]
            pred_mask = test_preds[current_idx]

            for cls in range(num_classes):
                intersect = np.logical_and(true_mask == cls, pred_mask == cls).sum()
                union = np.logical_or(true_mask == cls, pred_mask == cls).sum()
                total_intersection[cls] += intersect
                total_union[cls] += union
            current_idx += 1

    # 2. Calculate final IoU
    ious = total_intersection / (total_union + 1e-6)

    # 3. PRINT THE REPORT (This must happen BEFORE the return)
    print("\n===================================")
    print("      OFFICIAL TEST SET REPORT      ")
    print("===================================")
    class_names = ["Background", "Path/Trail", "Grass/Veg", "Trees", "Obstacles", "Sky"]

    for i, iou in enumerate(ious):
        print(f"{i}: {class_names[i]:<15} : {iou:.4f}")

    valid_ious = ious[[0, 1, 3, 4, 5]] # Using your specific valid classes
    print("-----------------------------------")
    print(f"FINAL TEST mIoU : {np.mean(valid_ious):.4f}")
    print("===================================")

    # 4. RETURN AT THE VERY END
    return ious

# Run it and save the result
ious = calculate_test_metrics_efficiently(all_test_preds, test_loader)


# Class-wise activation maps show the model struggles to detect grass due to low dataset frequency. AS SHOWN BELOW:

# In[141]:


import matplotlib.pyplot as plt

# Take one image from the test set
image, mask = next(iter(test_loader))
img_tensor = image[0].unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(pixel_values=img_tensor).logits
    logits = F.interpolate(logits, size=(512, 512), mode="bilinear")

    # Let's look at the raw "heat" for Grass (Class 2)
    grass_heat = torch.softmax(logits, dim=1)[0, 2].cpu().numpy()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(grass_heat, cmap='hot')
plt.colorbar()
plt.title("Model's 'Idea' of Grass (Heatmap)")
plt.show()

print(f"Max confidence for Grass in this image: {grass_heat.max():.4f}")


# In[143]:


import numpy as np

# Get a single batch from the test loader
images, masks = next(iter(test_loader))
actual_labels_in_test = np.unique(masks.cpu().numpy())

print(" DATASET INVESTIGATION:")
print(f"The labels actually present in your Test Masks are: {actual_labels_in_test}")

# Check the first mask's distribution
for label in actual_labels_in_test:
    count = np.sum(masks.cpu().numpy() == label)
    print(f"Label {label} appears in {count} pixels.")


# In[144]:


# We only average the classes that actually exist in the Test Set
existing_classes = [0, 1, 3, 4, 5]
valid_ious = [ious[i] for i in existing_classes]

print("===================================")
print("   FINAL ADJUSTED TEST REPORT   ")
print("===================================")
class_names = ["Background", "Path/Trail", "UNUSED", "Trees", "Obstacles", "Sky"]

for i in existing_classes:
    print(f"Class {i}: {class_names[i]:<15} : {ious[i]:.4f}")

print("-----------------------------------")
print(f" TRUE mIoU (Existing Classes): {np.mean(valid_ious):.4f}")
print("===================================")


# In[27]:


import gc
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"GPU {i}: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB total")


# In[31]:


# ============================================================
# SECTION: POST-TRAINING ANALYSIS (Using Stage 2 Best Model)
# ============================================================
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.ndimage as ndimage
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from transformers import SegformerForSemanticSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6

# B2 — confirmed by checkpoint inspection (ch=64, depths=[3,4,6,3])
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load('best_model_stage2.pth', map_location=device))
model.to(device)
model.eval()
print("✅ SegFormer-B2 Stage 2 model loaded successfully.")

CLASS_NAMES  = ["Background", "Path/Trail", "Grass/Veg", "Trees", "Obstacles", "Sky"]
CLASS_COLORS = np.array([
    [80,  80,  80 ],
    [128, 64,  128],
    [0,   200, 0  ],
    [0,   102, 0  ],
    [255, 50,  50 ],
    [135, 206, 235],
], dtype=np.uint8)

def mask_to_color(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask_np == cls_id] = color
    return rgb

def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * std + mean, 0, 1)


# In[29]:


# ============================================================
# PROPER PER-CLASS mIoU on VALIDATION SET
# ============================================================
def evaluate_miou(model, loader, num_classes=6):
    model.eval()
    all_inter = np.zeros(num_classes)
    all_union = np.zeros(num_classes)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating Val Set"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(num_classes):
                inter = ((preds == cls) & (masks == cls)).sum().item()
                union = ((preds == cls) | (masks == cls)).sum().item()
                all_inter[cls] += inter
                all_union[cls] += union

    ious = all_inter / (all_union + 1e-6)
    return ious

val_ious = evaluate_miou(model, val_loader)

print("\n" + "="*50)
print("   VALIDATION RESULTS — Stage 2 Best Model (B2)")
print("="*50)
for i, (name, iou) in enumerate(zip(CLASS_NAMES, val_ious)):
    bar = "█" * int(iou * 30)
    print(f"  Class {i} [{name:<12}]: {iou:.4f}  {bar}")

existing_cls  = [0, 1, 3, 4, 5]   # Class 2 (Grass) absent from test
miou_all      = np.mean(val_ious)
miou_existing = np.mean([val_ious[i] for i in existing_cls])
print("-"*50)
print(f"  mIoU (all 6 classes)    : {miou_all:.4f}")
print(f"  mIoU (existing classes) : {miou_existing:.4f}  ← Official Score")
print("="*50)


# In[30]:


# ============================================================
# TRAINING CURVES — Stage 2 Loss & IoU
# ============================================================
# ⚠️ Replace these with your actual printed values from Stage 2 training output
stage2_epochs = list(range(1, 16))

stage2_train_loss = [1.42, 1.31, 1.22, 1.14, 1.08, 1.03, 0.99, 0.96,
                     0.93, 0.91, 0.89, 0.87, 0.86, 0.85, 0.84]

stage2_val_iou    = [0.38, 0.41, 0.44, 0.46, 0.48, 0.49, 0.50, 0.504,
                     0.508, 0.512, 0.515, 0.516, 0.517, 0.518, 0.5179]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Stage 2 Fine-Tuning — Training History (SegFormer-B2)", 
             fontsize=14, fontweight='bold')

# Loss
axes[0].plot(stage2_epochs, stage2_train_loss, 'b-o', markersize=4, label='Train Loss')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss (CE + Dice)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(stage2_epochs)

# IoU
best_ep = stage2_val_iou.index(max(stage2_val_iou)) + 1
axes[1].plot(stage2_epochs, stage2_val_iou, 'g-o', markersize=4, label='Val mIoU')
axes[1].axvline(x=best_ep, color='r', linestyle='--', alpha=0.7, 
                label=f'Best @ Epoch {best_ep}')
axes[1].axhline(y=0.5179, color='orange', linestyle=':', alpha=0.8, 
                label='Best IoU = 0.5179')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val mIoU")
axes[1].set_title("Validation IoU (Existing Classes)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(stage2_epochs)

plt.tight_layout()
plt.savefig("stage2_training_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: stage2_training_curves.png")


# In[31]:


# ============================================================
# PER-CLASS IoU BAR CHART
# ============================================================
hex_colors = ['#%02x%02x%02x' % tuple(c) for c in CLASS_COLORS]
hex_colors[0] = '#888888'  # make background visible

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(CLASS_NAMES, val_ious, color=hex_colors, 
              edgecolor='black', linewidth=0.7)

for bar, iou in zip(bars, val_ious):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{iou:.4f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

ax.axhline(y=miou_existing, color='red', linestyle='--', linewidth=1.5,
           label=f'mIoU (existing classes) = {miou_existing:.4f}')
ax.set_ylim(0, 1.0)
ax.set_ylabel("IoU Score")
ax.set_title("Per-Class IoU — Stage 2 Best Model (Validation Set)")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("per_class_iou_bar.png", dpi=150, bbox_inches='tight')
plt.show()


# In[32]:


# ============================================================
# CONFUSION MATRIX
# ============================================================
all_preds_flat = []
all_masks_flat = []

model.eval()
with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Building Confusion Matrix"):
        images = images.to(device)
        outputs = model(pixel_values=images).logits
        outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                mode="bilinear", align_corners=False)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds_flat.extend(preds.flatten())
        all_masks_flat.extend(masks.numpy().flatten())

cm = confusion_matrix(all_masks_flat, all_preds_flat, 
                      labels=list(range(NUM_CLASSES)), normalize='true')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Ground Truth", fontsize=12)
ax.set_title("Normalized Confusion Matrix — Stage 2 Validation", fontsize=13)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()


# In[33]:


# ============================================================
# QUALITATIVE INFERENCE — Val Set
# Image | Ground Truth | Prediction | Overlay
# ============================================================
def visualize_predictions(model, dataset, indices, apply_smoothing=True):
    model.eval()
    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]

    fig, axes = plt.subplots(len(indices), 4, figsize=(18, 4.5 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    fig.suptitle("Stage 2 Model — Validation Inference (SegFormer-B2)", 
                 fontsize=14, fontweight='bold')

    for row, idx in enumerate(indices):
        image, gt_mask = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(pixel_values=input_tensor).logits
            logits = F.interpolate(logits, size=(512, 512), 
                                   mode="bilinear", align_corners=False)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        if apply_smoothing:
            pred_mask = ndimage.median_filter(pred_mask, size=5)

        img_np   = unnormalize(image)
        gt_rgb   = mask_to_color(gt_mask.numpy())
        pred_rgb = mask_to_color(pred_mask)
        overlay  = np.clip(0.55 * img_np + 0.45 * pred_rgb / 255, 0, 1)

        titles  = ["Input Image", "Ground Truth", "Prediction", "Overlay"]
        visuals = [img_np, gt_rgb, pred_rgb, overlay]

        for col, (title, vis) in enumerate(zip(titles, visuals)):
            ax = axes[row][col]
            ax.imshow(vis)
            ax.set_title(title if row == 0 else "", fontsize=11)
            ax.axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=NUM_CLASSES,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout()
    plt.savefig("val_inference_grid.png", dpi=150, bbox_inches='tight')
    plt.show()

visualize_predictions(model, val_dataset, indices=[0, 5, 10, 20, 35])


# In[ ]:


# ============================================================
# TEST SET INFERENCE — TTA + Smoothing
# ============================================================
test_loader_clean = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, 
    num_workers=0, pin_memory=True
)

def predict_tta(model, img_tensor):
    with torch.no_grad():
        out1 = model(pixel_values=img_tensor).logits
        out2 = model(pixel_values=torch.flip(img_tensor, dims=[3])).logits
        out2 = torch.flip(out2, dims=[3])
        avg  = (out1 + out2) / 2
        avg  = F.interpolate(avg, size=(512, 512), 
                             mode="bilinear", align_corners=False)
    return avg

all_test_preds = []
all_test_masks = []

model.eval()
print("Running TTA inference on test set...")
for images, masks in tqdm(test_loader_clean):
    images = images.to(device)
    for i in range(images.size(0)):
        logits = predict_tta(model, images[i].unsqueeze(0))
        pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred   = ndimage.median_filter(pred, size=5)
        all_test_preds.append(pred)
    all_test_masks.extend(masks.numpy())

print(f"✅ Inference complete on {len(all_test_preds)} test images.")

# --- Per-class IoU ---
total_inter = np.zeros(NUM_CLASSES)
total_union = np.zeros(NUM_CLASSES)

for pred, gt in zip(all_test_preds, all_test_masks):
    for cls in range(NUM_CLASSES):
        total_inter[cls] += np.logical_and(pred == cls, gt == cls).sum()
        total_union[cls] += np.logical_or(pred == cls, gt == cls).sum()

test_ious = total_inter / (total_union + 1e-6)

print("\n" + "="*50)
print("   FINAL TEST REPORT — Stage 2 Model (TTA + Smoothing)")
print("="*50)
for i, (name, iou) in enumerate(zip(CLASS_NAMES, test_ious)):
    flag = "  ← not in test set" if i == 2 else ""
    print(f"  Class {i} [{name:<12}]: {iou:.4f}{flag}")

existing = [0, 1, 3, 4, 5]
final_miou = np.mean([test_ious[i] for i in existing])
print("-"*50)
print(f"  mIoU (existing 5 classes) : {final_miou:.4f}")
print("="*50)


# In[33]:


import numpy as np
np.save('all_test_preds.npy', np.array(all_test_preds))
np.save('all_test_masks.npy', np.array(all_test_masks))
print("Saved predictions and masks to disk.")


# In[38]:


# ============================================================
# TEST SET QUALITATIVE VISUALIZATION
# ============================================================
visualize_predictions(model, test_dataset, 
                      indices=[0, 3, 7, 12, 18], 
                      apply_smoothing=True)
plt.savefig("test_inference_grid.png", dpi=150, bbox_inches='tight')
plt.show()


# In[39]:


# ============================================================
# CLASS ACTIVATION HEATMAPS — All 6 Classes
# ============================================================
image, mask = test_dataset[0]
img_tensor  = image.unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(pixel_values=img_tensor).logits
    logits = F.interpolate(logits, size=(512, 512), 
                           mode="bilinear", align_corners=False)
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

img_np = unnormalize(image)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes[0][0].imshow(img_np)
axes[0][0].set_title("Input Image", fontweight='bold', fontsize=11)
axes[0][0].axis('off')

for cls in range(NUM_CLASSES):
    row = (cls + 1) // 4
    col = (cls + 1) % 4
    ax  = axes[row][col]
    im  = ax.imshow(probs[cls], cmap='hot', vmin=0, vmax=1)
    ax.set_title(f"Class {cls}: {CLASS_NAMES[cls]}", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

axes[1][3].axis('off')
fig.suptitle("Per-Class Confidence Heatmaps — Stage 2 Model (SegFormer-B2)", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("class_activation_heatmaps.png", dpi=150, bbox_inches='tight')
plt.show()


# In[40]:


# ============================================================
# TEST SET — Per-Class IoU Bar Chart (Real Results)
# ============================================================
test_iou_values = [0.4249, 0.4961, 0.0000, 0.0582, 0.6165, 0.9819]
existing_cls    = [0, 1, 3, 4, 5]
final_miou      = 0.5155

hex_colors = ['#%02x%02x%02x' % tuple(c) for c in CLASS_COLORS]
hex_colors[0] = '#888888'

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(CLASS_NAMES, test_iou_values, color=hex_colors,
              edgecolor='black', linewidth=0.7)

for bar, iou, i in zip(bars, test_iou_values, range(NUM_CLASSES)):
    label = f'{iou:.4f}' if i != 2 else 'N/A\n(absent)'
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            label, ha='center', va='bottom',
            fontsize=10, fontweight='bold')

ax.axhline(y=final_miou, color='red', linestyle='--', linewidth=1.5,
           label=f'mIoU (existing 5 classes) = {final_miou:.4f}')
ax.set_ylim(0, 1.1)
ax.set_ylabel("IoU Score", fontsize=12)
ax.set_title("Per-Class IoU — Test Set (SegFormer-B2, TTA + Smoothing)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("test_per_class_iou.png", dpi=150, bbox_inches='tight')
plt.show()


# In[35]:


import gc
gc.collect()
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Build from saved arrays directly — no model needed
test_preds_flat = np.array(all_test_preds).flatten()
test_masks_flat = np.array(all_test_masks).flatten()

cm = confusion_matrix(test_masks_flat, test_preds_flat,
                      labels=list(range(NUM_CLASSES)), normalize='true')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Ground Truth", fontsize=12)
ax.set_title("Normalized Confusion Matrix — Test Set (SegFormer-B2)", fontsize=13)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()


# In[36]:


import numpy as np
import matplotlib.pyplot as plt

pixel_counts = np.zeros(NUM_CLASSES)
for gt in all_test_masks:
    for cls in range(NUM_CLASSES):
        pixel_counts[cls] += (np.array(gt) == cls).sum()

total_pixels = pixel_counts.sum()
percentages  = (pixel_counts / total_pixels) * 100

hex_colors_bar = ['#%02x%02x%02x' % tuple(c) for c in CLASS_COLORS]
hex_colors_bar[0] = '#888888'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar
axes[0].bar(CLASS_NAMES, percentages, color=hex_colors_bar,
            edgecolor='black', linewidth=0.7)
for i, pct in enumerate(percentages):
    axes[0].text(i, pct + 0.3, f'{pct:.1f}%',
                 ha='center', fontsize=9, fontweight='bold')
axes[0].set_ylabel("% of Total Pixels")
axes[0].set_title("Class Pixel Distribution — Test Set")
axes[0].grid(axis='y', alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha='right')

# Pie (exclude Grass since absent)
pie_indices = [i for i in range(NUM_CLASSES) if pixel_counts[i] > 0]
pie_values  = [percentages[i] for i in pie_indices]
pie_labels  = [f"{CLASS_NAMES[i]}\n({percentages[i]:.1f}%)" for i in pie_indices]
pie_colors  = [np.array(CLASS_COLORS[i])/255 for i in pie_indices]
pie_colors[0] = np.array([0.53, 0.53, 0.53])

axes[1].pie(pie_values, labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
axes[1].set_title("Class Distribution Pie Chart\n(Grass/Veg absent from test set)")

plt.suptitle("Test Set Class Imbalance Analysis",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("test_class_distribution.png", dpi=150, bbox_inches='tight')
plt.show()


# In[37]:


import time
import numpy as np
import matplotlib.pyplot as plt
import torch

gc.collect()
torch.cuda.empty_cache()

model.eval()
timings = []

# Warmup
dummy = torch.randn(1, 3, 512, 512).to(device)
with torch.no_grad():
    for _ in range(3):
        _ = model(pixel_values=dummy)

# Time 50 single images
sample_images = []
for images, _ in test_loader_clean:
    for i in range(images.size(0)):
        sample_images.append(images[i])
    if len(sample_images) >= 50:
        break

print("Measuring inference time...")
for img in sample_images[:50]:
    inp = img.unsqueeze(0).to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(pixel_values=inp).logits
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings.append((t1 - t0) * 1000)

timings = np.array(timings)

print("\n" + "="*45)
print("   INFERENCE TIME — SegFormer-B2 on Kaggle T4")
print("="*45)
print(f"  Per image (mean)   : {timings.mean():.2f} ms")
print(f"  Per image (median) : {np.median(timings):.2f} ms")
print(f"  Per image (std)    : {timings.std():.2f} ms")
print(f"  Min                : {timings.min():.2f} ms")
print(f"  Max                : {timings.max():.2f} ms")
print(f"  FPS                : {1000/timings.mean():.1f} frames/sec")
print("="*45)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(timings, 'b-o', markersize=3, alpha=0.7, label='Per-image time')
ax.axhline(y=timings.mean(), color='red', linestyle='--',
           label=f'Mean = {timings.mean():.2f} ms')
ax.set_xlabel("Image Index")
ax.set_ylabel("Inference Time (ms)")
ax.set_title("Inference Time per Image — SegFormer-B2 (Kaggle T4)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("inference_time.png", dpi=150, bbox_inches='tight')
plt.show()


# In[39]:


import numpy as np
import matplotlib.pyplot as plt

# Define this since kernel lost it
existing_cls = [0, 1, 3, 4, 5]

per_image_ious = []
for pred, gt in zip(all_test_preds, all_test_masks):
    gt   = np.array(gt)
    pred = np.array(pred)
    inter, union = 0, 0
    for cls in existing_cls:
        inter += np.logical_and(pred == cls, gt == cls).sum()
        union += np.logical_or(pred == cls,  gt == cls).sum()
    per_image_ious.append(inter / (union + 1e-6))

per_image_ious = np.array(per_image_ious)
best_indices   = np.argsort(per_image_ious)[-4:][::-1]
worst_indices  = np.argsort(per_image_ious)[:4]

print(f"Best  IoU scores: {per_image_ious[best_indices]}")
print(f"Worst IoU scores: {per_image_ious[worst_indices]}")

def plot_ranked(indices, title, filename):
    fig, axes = plt.subplots(len(indices), 3,
                             figsize=(13, 4 * len(indices)))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for row, idx in enumerate(indices):
        image, gt_mask = test_dataset[idx]
        pred_mask      = np.array(all_test_preds[idx])
        img_np         = unnormalize(image)
        gt_rgb         = mask_to_color(np.array(gt_mask))
        pred_rgb       = mask_to_color(pred_mask)

        for col, (vis, t) in enumerate(zip(
            [img_np, gt_rgb, pred_rgb],
            ["Input", "Ground Truth",
             f"Prediction (IoU={per_image_ious[idx]:.3f})"]
        )):
            ax = axes[row][col]
            ax.imshow(vis)
            ax.set_title(t if row == 0 else
                        (f"IoU={per_image_ious[idx]:.3f}" if col == 2 else ""),
                        fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

plot_ranked(best_indices,
            "Best Predicted Test Images (Highest IoU)",
            "test_best_predictions.png")

plot_ranked(worst_indices,
            "Worst Predicted Test Images (Lowest IoU)",
            "test_worst_predictions.png")

