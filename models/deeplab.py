import torch
import torch.nn as nn
import torchvision


def get_model(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights="DEFAULT"
    )

    model.classifier[-1] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1
    )

    return model