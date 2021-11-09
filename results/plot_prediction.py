import pandas as pd
from matplotlib import pyplot as plt
from ds.tensorboard import TensorboardExperiment
from PIL import Image
import os


def plot_prediction(predictions: str, k: int, exp: int, label: int, log_dir='runs/predictions',
                    root='..', data_dir='data/classification_20_clean'):
    tracker = TensorboardExperiment(log_path=log_dir+f'/k={k}_exp={exp}_label{label}')
    df = pd.read_csv(predictions)
    df = df[(df['k'] == k) & (df['experiment'] == exp) & (df['label'] == label)]
    df.reset_index(inplace=True)
    imgs = [Image.open(os.path.join(root, data_dir, img_path)) for img_path in df['path']]

    print(len(imgs))
    print(df.head())


def main():
    plot_prediction('predictions.csv', 1, 0, 1)


if __name__ == "__main__":
    main()
