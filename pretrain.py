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


LOG_PATH = "./runs"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data configuration
root = '.'
# root = '/content/drive/MyDrive/I2MTC' # for colab
samples_per_class = 100
data_dir = f"data/sim2arctic_{samples_per_class}"
model_name = 'resnet50'

stage = 'pretrain'
dataset = 'iNaturalist'
loss = torch.nn.CrossEntropyLoss(reduction="mean")

# Hyperparameters
LR = 1e-5  # lr < 5e-4
batch_size_train = 16
batch_size_test = 16

# experiment settings
EPOCH_COUNT = 50


def main():
    # create_sim2arctic_from_inaturalist(new_data_dir=data_dir, class_size=200) # use when data/sim2lcr is not created
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

    batch_tensor = next(iter(train_loader))[0]
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=4)
    plt.imshow(inv_normalize(grid_img).permute(1, 2, 0))
    plt.show()
    #
    # Model and Optimizer
    model = Model(model_name, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoint = None
    path = f'results/pretrain/acc= {0.57}.pth'
    checkpoint = torch.load(path)
    if checkpoint is None:
        pass
    else:
        print(f'Load checkpoint for model and optimizer from {path}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # TODO: scheduler =

    # Create the runners
    train_runner = Runner(train_loader, model, device, optimizer, loss=loss)
    test_runner = Runner(test_loader, model, device, loss=loss)

    # Run the epochs
    folder_save = f'results/pretrain'
    train_model(test_runner, train_runner, EPOCH_COUNT, tracker, folder_save)

    tracker.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, EPOCH_COUNT)
    tracker.add_hparams({'batch_size': batch_size_train, 'lr': LR, 'epochs': EPOCH_COUNT, 'samples/class': samples_per_class},
                        {'train_accuracy': train_runner.avg_accuracy,
                         'test_accuracy': test_runner.avg_accuracy,
                         'train_f1_score': train_runner.f1_score_metric,
                         'test_f1_score': test_runner.f1_score_metric
                         })

    # predictions_exp = pd.DataFrame({'k': k, 'experiment': experiment,
    #                                 'path': np.concatenate(test_runner.idxs),
    #                                 'label': np.concatenate(test_runner.y_true_batches),
    #                                 'prediction': np.concatenate(test_runner.y_pred_batches)})
    # predictions_exp = predictions_exp[predictions_exp['label'] != predictions_exp['prediction']]
    # predictions_exp['path'] = predictions_exp['path'].apply(lambda x: test.iloc[x, 1])
    # predictions = pd.concat([predictions, predictions_exp])
    # # print(predictions.head())
    #
    torch.cuda.empty_cache()
    #
    # predictions.to_csv('results/false_predictions.csv', index=False)


if __name__ == "__main__":
    main()
