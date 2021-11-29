import numpy as np
import pandas as pd
import torch

from ds.models import Model
from ds.runner import Runner, train_model, eval_model
from ds.tensorboard import TensorboardExperiment

from ds.load_data import get_data, train_test_split_k_shot, get_transforms
from ds.dataset import create_data_loader
import datetime
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data configuration
root = '.'
root_save = '.'
data_dir = "data/classification_17_clean"
model_name = 'resnet50'
stage = 'check_part_pretrain'  # 'check_full_pretrain', 'check_part_pretrain', 'no_pretrain'
dataset = 'arctic'
loss = torch.nn.CrossEntropyLoss(reduction="mean")

# Hyperparameters
LR = 1e-4  # lr < 5e-4
batch_size_train = 16
batch_size_test = 16

# experiment settings
EPOCH_COUNT = 30

# for colab
colab = True
save_checkpoint = False
if colab:
    print('Running on colab')
    batch_size_train = 32
    batch_size_test = 32
    root = '/content'
    data_dir = 'classification_17_clean'
    root_save = '/content/drive/MyDrive/I2MTC/'

LOG_PATH = f"{root_save}/runs"
folder_save = f'{root_save}/results/{stage}'

DO_TRAIN = True
kk = range(0, 8)
number_of_exp = 1


def main():
    torch.manual_seed(17)
    df, num_classes = get_data(root, data_dir)
    predictions = pd.DataFrame()

    for k in kk:
        if k == 0:
            DO_TRAIN = False
        else:
          DO_TRAIN = True
        for experiment in range(number_of_exp):
            print('*' * 10)
            print(f'k={k}, experiment={experiment}')
            print('*' * 10)
            # Setup the experiment tracker
            name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
            tracker = TensorboardExperiment(log_path=f'{LOG_PATH}/{stage}/experiments/k={k}_exp#{experiment}_lr={LR}/'
                                                     f'{name_time}')
            # Create the data loaders
            train, test = train_test_split_k_shot(df, k, experiment)
            train.to_csv(f'{root_save}/splits/train_k{k}_#{experiment}', index=False)
            test.to_csv(f'{root_save}/splits/test_k{k}_#{experiment}', index=False)
            transform = get_transforms(dataset)
            train_loader = create_data_loader(annotations_file=train, root=root, data_dir=data_dir,
                                              transform=transform['train'], batch_size=batch_size_train)
            test_loader = create_data_loader(annotations_file=test, root=root, data_dir=data_dir,
                                             transform=transform['test'], batch_size=batch_size_test)

            # Model and Optimizer
            model = Model(model_name, num_classes)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            checkpoint = None
            path = None

            if stage == 'no_pretrain':
                pass
            elif stage == 'check_part_pretrain':
                path = f'{root_save}/results/pretrain/acc= 0.60.pth'
                checkpoint = torch.load(path)
                print(f'Load checkpoint for model and optimizer from {path}')
                model.load_state_dict(checkpoint['model_state_dict'])
            elif stage == 'check_full_pretrain':
                path = 'results/pretrain/BBN.iNaturalist2018.res50.180epoch.best_model.pth'
                checkpoint = torch.load(path)
            scheduler = ReduceLROnPlateau(optimizer, 'min')
            # Create the runners
            train_runner = Runner(train_loader, model, device, optimizer, scheduler, loss=loss)
            test_runner = Runner(test_loader, model, device, loss=loss)

            # Run the epochs
            if DO_TRAIN:
                train_model(test_runner, train_runner, EPOCH_COUNT, tracker, folder_save, save_checkpoint=False)
                tracker.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, EPOCH_COUNT)
                tracker.add_hparams({'stage': stage, 'k': k, '#_of_exp': experiment, 'batch_size': batch_size_train,
                                     'epochs': EPOCH_COUNT, 'lr': LR},
                                    {'train_accuracy': train_runner.avg_accuracy,
                                     'test_accuracy': test_runner.avg_accuracy,
                                     'train_f1_score': train_runner.f1_score_metric,
                                     'test_f1_score': test_runner.f1_score_metric
                                     })
            else:
                eval_model(test_runner, tracker)
                tracker.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, 0)

    #         predictions_exp = pd.DataFrame({'k': k, 'experiment': experiment,
    #                                        'path': np.concatenate(test_runner.idxs),
    #                                         'label': test_runner.y_true_batches,
    #                                         'prediction': test_runner.y_pred_batches})
    #         predictions_exp = predictions_exp[predictions_exp['label'] != predictions_exp['prediction']]
    #         predictions_exp['path'] = predictions_exp['path'].apply(lambda x: test.iloc[x, 0])
    #         predictions = pd.concat([predictions, predictions_exp])
    #         # print(predictions.head())
    #
    #         torch.cuda.empty_cache()
    #
    # predictions.to_csv(f'{root_save}/results/false_predictions_17.csv', index=False)


if __name__ == "__main__":
    main()
