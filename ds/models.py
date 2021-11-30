import torch
from torchvision import models
import torch.nn as nn
from ds.checkpoints.BBN_master.lib.config import cfg
from ds.checkpoints.BBN_master.lib.net.network import Network


class Model(torch.nn.Module):
    def __init__(self, name: str, num_classes: int, stage: str, device, path: str = None):
        super().__init__()
        if stage == 'check_full_pretrain':
            full_inat_num_classes = 8142
            model_ft = Network(cfg, mode="test", num_classes=full_inat_num_classes)
            model_ft.load_model(path)
            model_ft.freeze_backbone()
            model_ft.classifier = nn.Sequential(nn.Linear(4096, 1000), nn.Linear(1000, 17))
        else:
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
            if stage == 'check_part_pretrain':
                checkpoint = torch.load(path)
                print(f'Load checkpoint for model and optimizer from {path}')
                model_ft.load_state_dict(checkpoint['model_state_dict'])

        self.model = model_ft
        self.model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
