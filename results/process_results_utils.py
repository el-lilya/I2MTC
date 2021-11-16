import pandas as pd
from matplotlib import pyplot as plt
from ds.tensorboard import TensorboardExperiment
from PIL import Image
import os
import datetime
import math


def plot_false_predictions(predictions: str, k: int, exp: int, labels, log_dir='../runs',
                           root='..', data_dir='data/classification_20_clean'):
    name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
    tracker = TensorboardExperiment(log_path=log_dir + f'/false_preds/k={k}_exp#{exp}_labels={labels}/{name_time}')
    df = pd.read_csv(predictions)
    df = df[(df['k'] == k) & (df['experiment'] == exp) & (df['label'].isin(labels))]
    df.reset_index(inplace=True)
    imgs = [Image.open(os.path.join(root, data_dir, img_path)) for img_path in df['path']]
    ncols = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure(figsize=(8, 6))
    st = fig.suptitle(f"True label : False predictions", fontsize="x-large")
    for i, img in enumerate(imgs):
        ax = plt.subplot(len(imgs) // ncols + 1, ncols, i + 1)
        ax.set_title(f'{df.label[i]}:{df.prediction[i]}')
        ax.axis('off')
        ax.set_aspect('equal')
        plt.imshow(img)
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    tracker.add_figure(f'PREDICTIONS', fig)
    tracker.flush()


def plot_train_images(k: int, exp: int, labels, log_dir='../runs',
                      root='..', split_dir='splits', data_dir='data/classification_20_clean'):
    name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
    tracker = TensorboardExperiment(log_path=log_dir + f'/train_images/k={k}_exp#{exp}_labels={labels}/{name_time}')
    df = pd.read_csv(os.path.join(root, split_dir, f'train_k{k}_#{exp}'))
    # df = df.sort_values('label')
    df = df[df['label'].isin(labels)]
    df.reset_index(inplace=True)
    imgs = [Image.open(os.path.join(root, data_dir, img_path)) for img_path in df['img_path']]
    ncols = 5
    fig = plt.figure(figsize=(8, 6))
    st = fig.suptitle(f"Images in train dataset for k={k}, exp={exp}, labels={labels}", fontsize="x-large")
    for i, img in enumerate(imgs):
        ax = plt.subplot(len(imgs) // ncols + 1, ncols, i + 1)
        ax.set_title(df.loc[i]['label'])
        ax.axis('off')
        ax.set_aspect('equal')
        plt.imshow(img)
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    # plt.show()
    tracker.add_figure(f'TRAIN IMAGES', fig)
    tracker.flush()


def get_mean_std_acc(name: str, name_csv: str, log_dir='../runs'):
    results_full = pd.read_csv(name)
    results = pd.DataFrame({'k': results_full['k'].unique()})
    dict_mean_acc = results_full.groupby('k')['test_accuracy'].mean().to_dict()
    dict_std_acc = results_full.groupby('k')['test_accuracy'].std().to_dict()
    results['mean_acc'] = results['k'].map(dict_mean_acc)
    results['std_acc'] = results['k'].map(dict_std_acc)
    # print(results)
    results.to_csv(name_csv)

    max_k = results['k'].max()
    name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
    tracker = TensorboardExperiment(log_path=log_dir + f'/accuracy/max_k={max_k}/{name_time}')
    fig = plt.figure(figsize=(8, 6))
    plt.plot(results['k'], results['mean_acc'])
    plt.fill_between(results['k'], (results['mean_acc'] - results['std_acc']),
                     (results['mean_acc'] + results['std_acc']), color='blue', alpha=0.1)
    plt.title('Mean accuracy')
    plt.xlabel('k')
    plt.ylabel('Mean accuracy')
    plt.grid()
    tracker.add_figure(f'ACCURACY', fig)
    tracker.flush()


def main():
    pass


if __name__ == "__main__":
    main()