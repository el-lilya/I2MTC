import numpy as np
import pandas as pd
import torch

from ds.models import Model
from ds.runner import Runner, train_model
from ds.tensorboard import TensorboardExperiment

from ds.load_data import get_data, train_test_split_k_shot, get_transforms
from ds.dataset import create_data_loader
import datetime
from ds.load_data import create_sim2arctic_from_inaturalist
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import os
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data configuration
root = '.'
samples_per_class = 50
data_dir = f"data/sim2arctic_{samples_per_class}"
model_name = 'resnet50'
folder_save = f'{root}/results/pretrain'
stage = 'pretrain'
dataset = 'iNaturalist'
loss = torch.nn.CrossEntropyLoss(reduction="mean")

# Hyperparameters
LR = 1e-4  # lr < 5e-4
batch_size_train = 16
batch_size_test = 16

# experiment settings
EPOCH_COUNT = 50
LOG_PATH = f"{root}/runs"

# for colab
colab = True
save_checkpoint = True
if colab:
    batch_size_train = 64
    batch_size_test = 64
    root = '/content'
    data_dir = 'sim2arctic_50'
    root_save = '/content/drive/MyDrive/I2MTC/'
    LOG_PATH = f"{root_save}/runs"
    folder_save = f'{root_save}/results/pretrain'

DO_TRAIN = True


def main():
    os.makedirs(folder_save, exist_ok=True)
    # create_sim2arctic_from_inaturalist(new_data_dir=data_dir, class_size=samples_per_class)
    df, num_classes = get_data(root, data_dir, img_format='.jpg')
    # Setup the experiment tracker
    name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
    tracker = TensorboardExperiment(log_path=LOG_PATH + f'/{stage}/experiments/{samples_per_class}_{EPOCH_COUNT}_'
                                                        f'{np.log10(LR)}_{name_time}')

    # # Create the data loaders
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df['label'])

    transform = get_transforms(dataset)
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    train_loader = create_data_loader(annotations_file=train, root=root, data_dir=data_dir,
                                      transform=transform['train'], batch_size=batch_size_train)
    test_loader = create_data_loader(annotations_file=test, root=root, data_dir=data_dir,
                                     transform=transform['test'], batch_size=batch_size_test)

    # batch_tensor = next(iter(train_loader))[0]
    # grid_img = torchvision.utils.make_grid(batch_tensor, nrow=4)
    # plt.imshow(inv_normalize(grid_img).permute(1, 2, 0))
    # plt.show()

    # Model and Optimizer
    model = Model(model_name, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoint = None
    path_load = f'{folder_save}/acc= {0.58}.pth'
    checkpoint = torch.load(path_load)
    if checkpoint is None:
        pass
    else:
        print(f'Loading checkpoint for model and optimizer from {path_load}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # Create the runners
    train_runner = Runner(train_loader, model, device, optimizer, scheduler, loss=loss)
    test_runner = Runner(test_loader, model, device, loss=loss)

    # Run the epochs
    if DO_TRAIN:
        train_model(test_runner, train_runner, EPOCH_COUNT, tracker, folder_save, save_checkpoint=False)

        tracker.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, EPOCH_COUNT)
        tracker.add_hparams({'batch_size': batch_size_train, 'lr': LR, 'epochs': EPOCH_COUNT,
                             'samples/class': samples_per_class},
                            {'train_accuracy': train_runner.avg_accuracy,
                             'test_accuracy': test_runner.avg_accuracy,
                             'train_f1_score': train_runner.f1_score_metric,
                             'test_f1_score': test_runner.f1_score_metric
                             })

        torch.cuda.empty_cache()
    if colab:
        path = f'{folder_save}/acc={test_runner.avg_accuracy: .2f}.pth'
        if test_runner.avg_accuracy > 0.5 and save_checkpoint:
            torch.save({
                'epoch': EPOCH_COUNT,
                'model_state_dict': train_runner.model.state_dict(),
                'optimizer_state_dict': train_runner.optimizer.state_dict(),
                'loss': test_runner.avg_loss
            }, path)
            print(f'Saved to {path}!')


if __name__ == "__main__":
    main()
