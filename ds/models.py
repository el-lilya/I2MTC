import torch
from torchvision import models
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        if name == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
            ct = 0
            for child in model_ft.children():
                ct += 1
                # print(ct, child)
                if ct < 8:
                    for param in child.parameters():
                        param.requires_grad = False
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            self.model = model_ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
