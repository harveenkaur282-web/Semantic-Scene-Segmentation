import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .model import get_segformer_model
from .data_loader import get_transforms
import pydensecrf.densecrf as dcrf  # Assume CRF library

def test_time_augmentation(model, image):
    # Simplified TTA: horizontal flip
    device = next(model.parameters()).device
    image = image.to(device)
    pred1 = model(image).logits
    pred2 = model(torch.flip(image, [3])).logits
    pred2 = torch.flip(pred2, [3])
    return (pred1 + pred2) / 2

def apply_crf(image, pred):
    # Simplified CRF post-processing
    crf = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 6)  # Adjust classes
    # Add unary and pairwise potentials (expand as needed)
    crf.setUnaryEnergy(-np.log(pred))
    crf.addPairwiseGaussian(sxy=3, compat=3)
    crf.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    return crf.inference(5)

def visualize_prediction(image, mask, pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[1].imshow(mask.cpu().numpy(), cmap='gray')
    ax[2].imshow(pred, cmap='gray')
    plt.show()

if __name__ == "__main__":
    model = get_segformer_model()
    model.load_state_dict(torch.load('path/to/model.pth'))  # Load trained model
    model.eval()
    transform = get_transforms(train=False)
    image = Image.open('path/to/test/image.jpg').convert('RGB')
    image_tensor = transform(image=image)['image'].unsqueeze(0)
    pred = test_time_augmentation(model, image_tensor)
    pred = apply_crf(image_tensor.squeeze(0).cpu().numpy(), pred.squeeze(0).cpu().numpy())
    visualize_prediction(image_tensor.squeeze(0), torch.zeros_like(pred), pred)