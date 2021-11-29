from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ds.metrics import Metric
from ds.tracking import ExperimentTracker, Stage
from typing import List
from ds.tensorboard import TensorboardExperiment


class Runner:
    def __init__(
            self,
            loader: DataLoader[Any],
            model: torch.nn.Module,
            device,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler=None,
            loss: torch.nn.modules.loss = torch.nn.CrossEntropyLoss(reduction="mean"),
    ) -> None:
        self.run_count = 0
        self.loader = loader
        self.accuracy_metric = Metric()
        self.loss = Metric()
        self.f1_score_metric = 0
        self.model = model
        self.optimizer = optimizer
        # Objective (loss) function
        self.compute_loss = loss
        self.y_true_batches: List[List[Any]] = []
        self.y_pred_batches: List[List[Any]] = []
        self.idxs: List[List[Any]] = []
        # Assume Stage based on presence of optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN
        self.device = device
        self.scheduler = scheduler

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average

    @property
    def avg_loss(self):
        return self.loss.average

    def run(self, desc: str, experiment: TensorboardExperiment):
        self.model.train(self.stage is Stage.TRAIN)

        # for x, y, idx in tqdm(self.loader, desc=desc, ncols=80):
        for x, y, idx in self.loader:
            loss, batch_accuracy = self._run_single(x.to(self.device), y.to(self.device))
            self.idxs += [idx.numpy()]
            # experiment.add_batch_metric("accuracy", batch_accuracy, self.run_count)

            if self.optimizer:
                # Reverse-mode AutoDiff (backpropagation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.y_true_batches = np.concatenate(self.y_true_batches)
        self.y_pred_batches = np.concatenate(self.y_pred_batches)

    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Validation Metrics
        y_np = y.cpu().detach().numpy()
        y_prediction_np = np.argmax(prediction.cpu().detach().numpy(), axis=1)
        batch_accuracy: float = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)
        self.loss.update(loss.item(), batch_size)

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]
        return loss, batch_accuracy

    def reset(self):
        self.accuracy_metric = Metric()
        self.loss = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []
        self.idxs = []


def run_epoch(
        test_runner: Runner,
        train_runner: Runner,
        experiment: TensorboardExperiment,
        epoch_id: int,
):
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    train_runner.f1_score_metric = f1_score(train_runner.y_true_batches,
                                            train_runner.y_pred_batches,
                                            average='macro')  # or 'weighted'
    # Log Training Epoch Loss and Metrics
    experiment.add_epoch_metric("loss", train_runner.avg_loss, epoch_id)
    # experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_metric("f1-score", train_runner.f1_score_metric, epoch_id)

    # Testing Loop
    experiment.set_stage(Stage.VAL)
    test_runner.run("Validation Batches", experiment)

    test_runner.f1_score_metric = f1_score(test_runner.y_true_batches,
                                           test_runner.y_pred_batches,
                                           average='weighted')  # or 'weighted'
    # Log Validation Epoch Loss and Metrics
    experiment.add_epoch_metric("loss", test_runner.avg_loss, epoch_id)
    # experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_metric("f1-score", test_runner.f1_score_metric, epoch_id)


def train_model(test_runner: Runner,
                train_runner: Runner,
                epochs: int,
                tracker: TensorboardExperiment,
                folder_save: Optional[str] = None,
                save_checkpoint: bool = False):
    for epoch_id in range(epochs):
        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        run_epoch(test_runner, train_runner, tracker, epoch_id)
        train_runner.scheduler.step(test_runner.avg_loss)
        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"\n[Epoch: {epoch_id + 1}/{epochs}]",
                f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print(summary)
        # Flush the tracker after every epoch for live updates
        tracker.flush()
        torch.cuda.empty_cache()


def eval_model(test_runner: Runner,
               tracker: TensorboardExperiment):
    tracker.set_stage(Stage.VAL)
    test_runner.run("Validation Batches", tracker)

    test_runner.f1_score_metric = f1_score(test_runner.y_true_batches,
                                           test_runner.y_pred_batches,
                                           average='weighted')  # or 'weighted'
    # Log Validation Epoch Loss and Metrics
    # tracker.add_epoch_metric("accuracy", test_runner.avg_accuracy, 0)
    tracker.add_epoch_metric("f1-score", test_runner.f1_score_metric, 0)
    summary = ", ".join(
        [
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
        ]
    )
    print(summary)
    tracker.flush()
    torch.cuda.empty_cache()
