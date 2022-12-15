"""
This module contains custom Callback implementations
"""
import itertools
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import confusion_matrix

from lowdataregime.classification.model.classification_module import ClassificationModule


class BaseCallback(Callback, ABC):
    """
    Base callback class
    """

    def on_init_start(self, trainer):
        """
        Callback called on init_start
        :param trainer: The trainer object
        """
        print(f"Callback {self.__class__.__name__} is set!")

    def on_general_epoch_end(self, trainer, pl_module, outputs, mode):
        """
        Callback called on every epoch end
        :param trainer: The trainer object
        :param pl_module: The Pytorch Lightning module object
        :param outputs: The outputs of the steps
        :param mode: The current device (train, valid, test)
        """
        pass

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        Callback called on training epoch end
        :param trainer: The trainer object
        :param pl_module: The Pytorch Lightning module object
        :param outputs: The outputs of the steps
        """
        self.on_general_epoch_end(trainer, pl_module, outputs, "train")
        super().on_train_epoch_end(trainer, pl_module, outputs)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        """
        Callback called on validation epoch end
        :param trainer: The trainer object
        :param pl_module: The Pytorch Lightning module object
        """
        # TODO: This callback has no output argument
        super().on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        """
        Callback called on test epoch end
        :param trainer: The trainer object
        :param pl_module: The Pytorch Lightning module object
        """
        # TODO: This callback has no output argument
        super().on_test_epoch_end(trainer, pl_module)


class ConfusionMatrixCallback(BaseCallback):
    """
    Callback class to plot the confusion matrix after every epoch
    """

    def __init__(self):
        self.title = None
        self.classes = None

    def setup(self, trainer, pl_module, stage: str):
        """
        Callback called on the setup phase
        :param trainer: The trainer object
        :param pl_module: The Pytorch Lightning module object
        :param stage: UNKNOWN
        """
        if not hasattr(trainer, 'datamodule'):
            raise ValueError("The trainer has no datamodule, please use the base DataModule")
        self.classes = trainer.datamodule.classes
        self.title = f"Confusion of model {pl_module.name}"

    def plot_confusion(self, targets, logits):
        """
        Method to plot the confusion matrix
        :param targets: The targets of the prediction
        :param logits: The predicted logits
        :return: Matplotlib figure
        """

        pred = torch.argmax(logits, dim=-1)
        confusion = confusion_matrix(pred, targets, num_classes=len(self.classes)).cpu().numpy()

        # Normalize
        prediction_sum = confusion.sum(axis=1)
        confusion = confusion / prediction_sum[:, None]

        figure = plt.figure(figsize=(8, 8))
        axes = plt.gca()

        figure.suptitle(self.title)

        axes.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)

        tick_marks = np.arange(len(self.classes))
        plt.setp(
            axes,
            xticks=tick_marks,
            xticklabels=self.classes,
            yticks=tick_marks,
            yticklabels=self.classes,
        )

        # Use white text if squares are dark; otherwise black.
        threshold = 0.5
        for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
            color = "white" if confusion[i, j] > threshold else "black"
            axes.text(j, i, confusion[i, j], horizontalalignment="center", color=color)
            axes.set_ylabel("True label")
            axes.set_xlabel("Predicted label")

        plt.tight_layout()
        figure.canvas.draw()

        return figure

    def on_general_epoch_end(self, trainer, pl_module, outputs, mode):
        """
        Plots the confusion matrix on each epoch end
        """
        logits = torch.stack([x[0][f"{mode}_logits"] for x in outputs[0]])
        targets = torch.stack([x[0][f"{mode}_targets"] for x in outputs[0]])

        fig = self.plot_confusion(targets, logits)
        pl_module.logger.experiment.add_figure(f"{mode}_confusion", fig,
                                               global_step=pl_module.current_epoch)


class HistogramCallback(BaseCallback):
    """
    Callback class to log the histogram of the model parameters
    """

    def on_fit_start(self, trainer, pl_module):
        for name, params in pl_module.named_parameters():
            pl_module.logger.experiment.add_histogram(name, params, pl_module.current_epoch)


class MetricsCallback(BaseCallback):
    """
    PyTorch Lightning metric callback.
    Source: https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py
    """

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module: ClassificationModule):
        """
        Stores the accuracy on the validation epoch end
        """
        self.metrics.append(pl_module.valid_acc.compute())
