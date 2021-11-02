import torch
from torchvision import models
import torch.nn as nn


class ModelBaseline(torch.nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        if name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
