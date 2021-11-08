import pandas as pd
from matplotlib import pyplot as plt
from ds.tensorboard import TensorboardExperiment
from PIL import Image


def plot_prediction(predictions: str, k: int, exp: int, label: int, log_dir='runs/predictions', root='../'):
    tracker = TensorboardExperiment(log_path=log_dir+f'/k={k}_exp={exp}_label{label}')
    df = pd.read_csv(predictions)
    df = df[(df['k'] == k) & (df['experiment'] == exp) & (df['label'] == label)]
    imgs = [Image.open(root+img_path) for img_path in df['path']]
    # Image.open('..\data/classification_20_clean/1/118.jpeg')
    print(len(imgs))
    print(df.head())


def main():
    plot_prediction('predictions.csv', 1, 0, 1)


if __name__ == "__main__":
    main()
