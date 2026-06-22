import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

def build_model(
        num_classes=7,
        dropout_rate=0.5):

    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V2
    )

    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(
            2048,
            num_classes
        )
    )

    return model