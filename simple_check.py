import pandas as pd
from ds.load_data import urls_from_clip, imgs_from_url, get_data
import os
from torchvision import models
import numpy as np
import timm
import torch


def main():
    # avail_pretrained_models = timm.list_models('*dense*', pretrained=True)
    # print(avail_pretrained_models)
    #
    # model_ft = timm.create_model('densenet121', pretrained=True, num_classes=10)
    # print(model_ft)
    model_specs = {
        "name": "inat2021_supervised",
        "display_name": "iNat2021 Supervised",
        "color": "C9",
        "format": PYTORCH,
        "backbone": RESNET50,
        "weights": PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_supervised_large.pth.tar',
        "training_dataset": INAT2021,
        "train_objective": SUPERVISED,
        "pretrained_weights": IMAGENET
    },


if __name__ == "__main__":
    main()
