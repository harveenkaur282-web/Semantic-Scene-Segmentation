import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OffRoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.class_mapping = {27: 4, 39: 5}  # Remap classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))  # Assume mask format
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply class remapping
        mask = torch.tensor(mask, dtype=torch.long)
        for old_class, new_class in self.class_mapping.items():
            mask[mask == old_class] = new_class
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask.numpy())
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# Albumentations transforms
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

if __name__ == "__main__":
    # Example usage
    dataset = OffRoadDataset('path/to/images', 'path/to/masks', transform=get_transforms(train=True))
    print(f"Dataset size: {len(dataset)}")