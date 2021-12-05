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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
import clip
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data configuration
root = '.'
root_save = '.'
data_dir = "data/classification_17_clean_clean"
model_name = 'RN50'
dataset = 'arctic'
loss = torch.nn.CrossEntropyLoss(reduction="mean")

# Hyperparameters
batch_size_train = 16
batch_size_test = 4
LR = 1e-4  # lr < 5e-4

# experiment settings
stage = 'clip'

# for colab
colab = False
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

# test train
k = 0
experiment = 0


def main():
    df, num_classes = get_data(root, data_dir)
    folder_save = f'{root_save}/results/{stage}'

    # Setup the experiment tracker
    name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
    tracker = TensorboardExperiment(log_path=f'{LOG_PATH}/{stage}/experiments/k={k}_exp#{experiment}_'
                                                 f'/{name_time}')
    # Create the data loaders
    model, preprocess = clip.load(model_name)
    model.to(device)
    model.eval()
    test_loader = create_data_loader(annotations_file=df, root=root, data_dir=data_dir,
                                     transform=preprocess, batch_size=batch_size_test)
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    # )
    # batch_tensor = next(iter(test_loader))[0]
    # grid_img = torchvision.utils.make_grid(batch_tensor, nrow=4)
    # plt.imshow(inv_normalize(grid_img).permute(1, 2, 0))
    # plt.show()
    descriptions = dict(enumerate(['empty slots', 'pepper plant', 'tomato plant', 'kohlrabi plant', 'frisee plant',
                                   'lettuce plant', 'mint plant', 'lettuce oakleaf plant', 'radish plant',
                                   'basil plant',
                                   'curly parsley plant', 'cress plant', 'chard plant', 'brassica plant',
                                   'lettuce endivia plant', 'parsley plant', 'chives plant']))
    text_tokens = clip.tokenize(["This is " + desc for desc in descriptions.values()])
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features.to(device)
    y_true_batches = []
    y_pred_batches = []
    n = 0
    for image_input, label, idx in test_loader:
        image_features = model.encode_image(image_input.to(device)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)
        y_np = label.cpu().detach().numpy()
        y_prediction_np = top_labels.cpu().detach().numpy()
        y_true_batches += [y_np]
        y_pred_batches += [y_prediction_np]
        n += 1
        if n > 1:
            break
    y_true_batches = np.concatenate(y_true_batches)
    y_pred_batches = np.concatenate(y_pred_batches)
    f1_score_metric = f1_score(y_true_batches, y_pred_batches, average='weighted')
    print(f1_score_metric)
    tracker.add_epoch_confusion_matrix(y_true_batches, y_pred_batches, 0)
    #
    tracker.add_hparams({'stage': stage, 'k': 0, '#_of_exp': 0, 'batch_size': 0,
                         'epochs': 0, 'lr': 0},
                        {'test_f1_score': f1_score_metric})


if __name__ == "__main__":
    main()
