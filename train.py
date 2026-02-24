import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import SegmentationDataset
from models.deeplab import get_model   # adjust if path slightly different

# ------------------------
# Config
# ------------------------
num_classes = 10
batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# Dataset
# ------------------------
train_image_dir = "dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images"
train_mask_dir = "dataset/Offroad_Segmentation_Training_Dataset/train/Segmentation"

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = SegmentationDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ------------------------
# Model
# ------------------------
model = get_model(num_classes)
model = model.to(device)

print("Model loaded successfully.")

images, masks = next(iter(train_loader))

print("Image shape:", images.shape)
print("Mask shape:", masks.shape)
print("Mask unique values:", torch.unique(masks))