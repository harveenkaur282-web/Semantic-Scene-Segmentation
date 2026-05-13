import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = F.resize(image, (256, 256), interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, (256, 256), interpolation=InterpolationMode.NEAREST)

        if self.transform is not None:
            image = self.transform(image)
        
        self.value_mapping = {100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

        mask = np.array(mask)
        mapped_mask = np.zeros_like(mask)

        for original_value, new_value in self.value_mapping.items():
            mapped_mask[mask == original_value] = new_value
            
        mask = torch.as_tensor(mapped_mask, dtype=torch.long)


        return image, mask