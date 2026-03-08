import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .data_loader import OffRoadDataset, get_transforms
from .model import get_segformer_model

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        # Simplified Dice (expand as needed)
        dice_loss = 1 - (2 * (pred * target).sum() + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Two-stage: Fine-tune with lower LR if needed
    optimizer = torch.optim.Adam(model.parameters(), lr=lr/10)
    for epoch in range(epochs//2):
        # Similar loop
        pass

if __name__ == "__main__":
    train_dataset = OffRoadDataset('path/to/train/images', 'path/to/train/masks', transform=get_transforms(train=True))
    val_dataset = OffRoadDataset('path/to/val/images', 'path/to/val/masks', transform=get_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    model = get_segformer_model()
    train_model(model, train_loader, val_loader)