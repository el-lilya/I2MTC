import pandas as pd
from ds.load_data import urls_from_clip, imgs_from_url, get_data
import os


def main():
    # urls_from_clip()
    # imgs_from_url()
    get_data(root='.', img_dir='data/clip/image_folder', img_format = '.jpg')

if __name__ == "__main__":
    main()
