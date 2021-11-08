from enum import Enum, auto
from pathlib import Path
# from typing import Protocol
from typing import List
from typing_extensions import Protocol



import numpy as np


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ExperimentTracker(Protocol):
    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(
        self, y_true: List[np.array], y_pred: List[np.array], step: int
    ):
        """Implements logging a confusion matrix at epoch-level."""
    def add_hparams(self, hparams: dict, metric_dict: dict):
        """Adds a set of hyperparameters to be compared in TensorBoard."""
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        """Add image data to summary."""
    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        """Render matplotlib figure into an image and add it to summary."""
