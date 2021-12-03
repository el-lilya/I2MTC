import torch
from torchvision import models
import torch.nn as nn
from ds.checkpoints.BBN_master.lib.config import cfg
from ds.checkpoints.BBN_master.lib.net.network import Network


class Model(torch.nn.Module):
    def __init__(self, name: str, num_classes: int, stage: str, device, path: str = None, k: int = None):
        super().__init__()
        if stage == 'check_full_pretrain':
            full_inat_num_classes = 8142
            model_ft = Network(cfg, mode="test", num_classes=full_inat_num_classes)
            model_ft.load_model(path)
            model_ft.freeze_backbone()
            if k < 4:
                model_ft.classifier = nn.Linear(4096, num_classes)
            else:
                model_ft.classifier = nn.Sequential(nn.Linear(4096, 1000), nn.Linear(1000, num_classes))

        elif stage in ['pretrain', 'check_part_pretrain']:
            model_ft = models.resnet50(pretrained=True)
            for param in model_ft.parameters():
                param.requires_grad = False
            if stage == 'pretrain':
                for param in model_ft.layer4.parameters():
                    param.requires_grad = True
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

        elif stage == 'no_pretrain':
            model_ft = models.resnet50(pretrained=True)
            for param in model_ft.parameters():
                param.requires_grad = False
            num_ftrs = model_ft.fc.in_features
            if k < 4:
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
            else:
                model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1000), nn.Linear(1000, num_classes))
        self.model = model_ft
        if stage == 'check_part_pretrain':
            checkpoint = torch.load(path)
            print(f'Load checkpoint for model and optimizer from {path}')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
