import pandas as pd
from ds.load_data import urls_from_clip, imgs_from_url, get_data
import os
from torchvision import models
import torch.nn as nn


def main():
    # urls_from_clip()
    # imgs_from_url()
    # print(get_data(root='.', img_dir='data/clip/image_folder', img_format='.jpg'))
    layer = nn.Linear(10, 20)
    for parameter in layer.parameters():
        parameter.requires_grad = False


if __name__ == "__main__":
    main()
