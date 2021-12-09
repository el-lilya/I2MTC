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
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
import clip
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# https://github.com/openai/CLIP.git
# Data configuration
root = '.'
root_save = '.'
data_dir = "data/classification_17_clean_clean"
model_name = 'resnet50'
dataset = 'arctic'
loss = torch.nn.CrossEntropyLoss(reduction="mean")

# Hyperparameters
batch_size_train = 64
batch_size_test = 64
# LR = 1e-4  # lr < 5e-4
LR = 1e-3  # lr < 5e-4

# experiment settings
# stage = 'check_part_pretrain'  # 'check_full_pretrain', 'check_part_pretrain', 'no_pretrain', 'check_clip'

# for colab
colab = True
save_checkpoint = False
if colab:
    print('Running on colab')
    batch_size_train = 64
    batch_size_test = 64
    root = '/content'
    data_dir = 'classification_17_clean_clean'
    root_save = '/content/drive/MyDrive/I2MTC/'

LOG_PATH = f"{root_save}/runs"
# folder_save = f'{root_save}/results/{stage}'

# # final train
# EPOCH_COUNT = 50
# kk = range(1, 6)
# number_of_exp = 5

# test train
EPOCH_COUNT = 50
# kk = list(range(0, 6)) + [8]
kk = [0]
number_of_exp = 0
comment = '5_warmup'
pretrain_dataset = None
# pretrain_dataset = 'iNaturalist'  # iNaturalist, clip
stage = 'check_part_pretrain'


def main():
    df, num_classes = get_data(root, data_dir)
    # predictions = pd.DataFrame()
    for pretrain_dataset in ['iNaturalist', 'clip']:
        folder_save = f'{root_save}/results/{stage}'
        for k in kk:
            if k == 0:
                do_train = False
            else:
                do_train = True
            for experiment in range(3, 3 + number_of_exp):
                print('*' * 10)
                print(f'k={k}, experiment={experiment}')
                print('*' * 10)
                # Setup the experiment tracker
                name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
                if stage == 'check_part_pretrain':
                    tracker = TensorboardExperiment(log_path=f'{LOG_PATH}/{stage}/{pretrain_dataset}/experiments/'
                                                             f'k={k}_exp#{experiment}_lr={LR}_{comment}/{name_time}')
                else:
                    tracker = TensorboardExperiment(log_path=f'{LOG_PATH}/{stage}_{model_name}/experiments/k={k}_exp#{experiment}_'
                                                             f'lr={LR}_{comment}/{name_time}')
                # Create the data loaders
                train, test = train_test_split_k_shot(df, k, experiment)
                train.to_csv(f'{root_save}/splits/train_k{k}_#{experiment}', index=False)
                test.to_csv(f'{root_save}/splits/test_k{k}_#{experiment}', index=False)
                if stage == 'check_clip':
                    preprocess = clip.load(model_name)[1]
                else:
                    preprocess = None
                transform = get_transforms(dataset, stage, preprocess)
                train_loader = create_data_loader(annotations_file=train, root=root, data_dir=data_dir,
                                                  transform=transform['train'], batch_size=batch_size_train)
                test_loader = create_data_loader(annotations_file=test, root=root, data_dir=data_dir,
                                                 transform=transform['test'], batch_size=batch_size_test)
                # inv_normalize = transforms.Normalize(
                #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                # )
                # batch_tensor = next(iter(train_loader))[0]
                # grid_img = torchvision.utils.make_grid(batch_tensor, nrow=4)
                # plt.imshow(inv_normalize(grid_img).permute(1, 2, 0))
                # plt.show()
                # Model and Optimizer
                path = None
                if stage in ['no_pretrain', 'check_clip']:
                    pass
                elif stage == 'check_part_pretrain':
                    path = f'{root_save}/results/pretrain/{pretrain_dataset}/acc= 0.54.pth'
                elif stage == 'check_full_pretrain':
                    path = f"{root}/BBN.iNaturalist2018.res50.180epoch.best_model.pth"
                else:
                    print(f'Stage {stage} is not implemented!')
                model = Model(model_name, num_classes, stage, device, path, k)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
                scheduler = StepLR(optimizer, step_size=15, gamma=0.2)
                scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5,
                                                          after_scheduler=scheduler)
                # Create the runners
                train_runner = Runner(train_loader, model, device, optimizer, scheduler_warmup, loss=loss)
                test_runner = Runner(test_loader, model, device, loss=loss)

                # Run the epochs
                if do_train:
                    train_model(test_runner, train_runner, EPOCH_COUNT, tracker, folder_save, save_checkpoint=False)
                    tracker.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, EPOCH_COUNT)

                else:
                    eval_model(test_runner, tracker)
                    tracker.add_epoch_confusion_matrix(test_runner.y_true_batches, test_runner.y_pred_batches, 0)

                tracker.add_hparams({'stage': f'{stage}_{pretrain_dataset}', 'k': k, '#_of_exp': experiment, 'batch_size': batch_size_train,
                                     'epochs': EPOCH_COUNT, 'lr': LR},
                                    {
                                        'train_f1_score': train_runner.f1_score_metric,
                                        'test_f1_score': test_runner.best_f1_score
                                    })
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
