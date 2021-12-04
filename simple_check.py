import pandas as pd
from ds.load_data import urls_from_clip, imgs_from_url, get_data
import os
from torchvision import models


def main():
    # urls_from_clip()
    # imgs_from_url()
    # print(get_data(root='.', img_dir='data/clip/sim2arctic_clip', img_format='.jpg'))
    for i, bottleneck in enumerate(models.resnet50().layer4.children()):
        if i > 1:
            print(i, bottleneck)


if __name__ == "__main__":
    main()
