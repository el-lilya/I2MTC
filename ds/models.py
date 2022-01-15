import torch
from torchvision import models
import torch.nn as nn
from ds.checkpoints.BBN_master.lib.config import cfg
from ds.checkpoints.BBN_master.lib.net.network import Network
import clip
import timm


class Model(torch.nn.Module):
    def __init__(self, name: str, num_classes: int, stage: str, device, path: str = None, k: int = None):
        super().__init__()
        self.stage = stage
        if stage == 'iNat2018_BBN':
            full_inat_num_classes = 8142
            model_ft = Network(cfg, mode="test", num_classes=full_inat_num_classes)
            model_ft.load_model(path)
            model_ft.freeze_backbone()
            model_ft.classifier = nn.Linear(4096, num_classes)
            # if k < 4:
            #     model_ft.classifier = nn.Linear(4096, num_classes)
            # else:
            #     model_ft.classifier = nn.Sequential(nn.Linear(4096, 1000), nn.Linear(1000, num_classes))
        elif stage == 'check_clip':
            model, preprocess = clip.load(name)
            for param in model.parameters():
                param.requires_grad = False
            encoder = model.visual
            hid_dim = 1000
            if 'RN' in name:
                print('encoder output_dim = ', encoder.output_dim)
                fc = nn.Sequential(nn.Linear(encoder.output_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, num_classes))
                # fc = nn.Linear(encoder.output_dim, num_classes)
                model_ft = nn.Sequential(encoder, fc)
            else:
                encoder.proj = None
                print('encoder width = ', encoder.transformer.width)
                fc = nn.Sequential(nn.Linear(encoder.transformer.width, hid_dim), nn.ReLU(),
                                   nn.Linear(hid_dim, num_classes))
                # fc = nn.Linear(encoder.transformer.width, num_classes)
                model_ft = nn.Sequential(encoder, fc)
            self.encoder = encoder
            self.fc = fc
        elif stage in ['pretrain', 'check_part_pretrain']:
            model_ft = models.resnet50(pretrained=True)
            for param in model_ft.parameters():
                param.requires_grad = False
            for i, bottleneck in enumerate(model_ft.layer4.children()):
                if i > 1:
                    for j, layer in enumerate(bottleneck.children()):
                        if j > 1:
                            for param in layer.parameters():
                                param.requires_grad = True
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            if stage == 'check_part_pretrain':
                checkpoint = torch.load(path)
                print(f'Load checkpoint for model and optimizer from {path}')
                model_ft.load_state_dict(checkpoint['model_state_dict'])

        elif stage == 'no_pretrain':
            # available names: resnet50, vit_base_patch16_224, densenet121
            # if name == 'resnet50':
            # model_ft = models.resnet50(pretrained=True)
            model_ft = timm.create_model(name, pretrained=True, num_classes=num_classes)
            for param in model_ft.parameters():
                param.requires_grad = False
            for param in model_ft.fc.parameters():
                param.requires_grad = True
        self.model = model_ft
        self.model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stage == 'check_clip':
            x = self.encoder(x.half())
            x = self.fc(x.float())
            return x
        else:
            return self.model(x)
