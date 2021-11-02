import torch
import os
import pandas as pd
from torchvision import transforms


IMAGE_SIZE = 224


def get_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((800, 1000)),  # remove bottom digits in some pictures
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    return data_transforms


def get_data(root: str, img_dir: str):
    df = pd.DataFrame()
    for label in os.listdir(os.path.join(root, img_dir)):
        df_i = pd.DataFrame(
            {'img_path': [x for x in os.listdir(os.path.join(root, img_dir, label)) if x.endswith('.jpeg')],
             'label': label})
        df_i['img_path'] = df_i['img_path'].apply(lambda x: os.path.join(img_dir, label, x))
        df = pd.concat([df, df_i])
    num_classes = df['label'].nunique()
    return df, num_classes


def train_test_split(df: pd.DataFrame, k: int, num_of_exp: int):
    num_classes = df.label.nunique()
    train = stratified_sample_df(df, 'label', k, num_of_exp)
    # indices = list(map(int, train.index))
    test = df.loc[[x for x in df.index if x not in train.index]]
    if ~torch.cuda.is_available():
        test = test.sample(k * num_classes, random_state=42)  # for cpu to train faster
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, test


def stratified_sample_df(df: pd.DataFrame, col: str, n_samples, random_state: int = 42):
    n = min(n_samples, df[col].value_counts().min())
    if n < n_samples:
        print('Too big K')
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=random_state))
    df_.index = df_.index.droplevel(0)
    return df_


