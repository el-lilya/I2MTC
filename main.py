import torch

from ds.models import ModelBaseline
from ds.runner import Runner, run_epoch
from ds.tensorboard import TensorboardExperiment

from ds.load_data import get_data, train_test_split, get_transforms
from ds.dataset import create_data_loader
import datetime

# c

# Hyperparameters
EPOCH_COUNT = 10
LR = 1e-4  # 5e-5
LOG_PATH = "./runs"
batch_size_train = 8
batch_size_test = 16
# Data configuration
# root = '/content/drive/MyDrive/I2MTC' # for colab
root = '.'
data_dir = "./data/classification_20_clean"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
baseline_name = 'resnet50'

# experiment settings
kk = [1, 3, 5]
number_of_exp = 5


def main():
    df, num_classes = get_data(root, data_dir)

    for k in kk:
        for experiment in range(number_of_exp):
            # Setup the experiment tracker
            name_time = datetime.datetime.now().strftime('%d%h_%I_%M')
            tracker = TensorboardExperiment(log_path=LOG_PATH+'/'+f'k={k}_exp#{experiment}/{name_time}')

            # Create the data loaders
            train, test = train_test_split(df, k, experiment)
            train.to_csv(f'./splits/train_k{k}_#{experiment}')
            transforms = get_transforms()
            train_loader = create_data_loader(annotations_file=train, root=root, transform=transforms['train'],
                                              batch_size=batch_size_train)
            test_loader = create_data_loader(annotations_file=test, root=root, transform=transforms['test'],
                                             batch_size=batch_size_test)

            # Model and Optimizer
            model = ModelBaseline(baseline_name, num_classes)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # Create the runners
            test_runner = Runner(test_loader, model, device)
            train_runner = Runner(train_loader, model, device, optimizer)

            # Run the epochs
            for epoch_id in range(EPOCH_COUNT):

                run_epoch(test_runner, train_runner, tracker, epoch_id)

                # Compute Average Epoch Metrics
                summary = ", ".join(
                    [
                        f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
                        f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                        f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
                    ]
                )
                print("\n" + summary + "\n")

                # Flush the tracker after every epoch for live updates
                tracker.add_hparams({'k': k, '#_of_exp': experiment, 'epochs': EPOCH_COUNT},
                                    {'train_accuracy': train_runner.avg_accuracy,
                                     'test_accuracy': test_runner.avg_accuracy})
                # Reset the runners
                train_runner.reset()
                test_runner.reset()
                tracker.flush()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()