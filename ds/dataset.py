from typing import Any
import os
from PIL import Image
import pandas as pd

from torch.utils.data import DataLoader, Dataset


class ArcticDataset(Dataset):
    # def __init__(self, annotations_file: pd.DataFrame, root: str, transform=None):
    def __init__(self, annotations_file: pd.DataFrame, root: str, data_dir="data/classification_20_clean",
                 transform=None):
        self.annotations_file = annotations_file
        self.root = root
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.data_dir, self.annotations_file.iloc[idx, 0])
        label = int(self.annotations_file.iloc[idx, 1])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label, idx


def create_data_loader(annotations_file: pd.DataFrame, root: str, data_dir: str, transform, batch_size: int,
                       shuffle: bool = True) -> DataLoader[Any]:
    dataset = ArcticDataset(annotations_file=annotations_file, root=root, data_dir=data_dir, transform=transform)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=0
                      )
