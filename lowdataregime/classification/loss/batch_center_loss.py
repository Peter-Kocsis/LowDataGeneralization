"""
Based on https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
"""
from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor

from lowdataregime.classification.loss.losses import LossDefinition, LossType
from lowdataregime.parameters.params import HyperParameterSet
from lowdataregime.utils.distance import DistanceDefinitionSet, EuclideanDistanceDefinitionSet, Distance


class BatchCenterLossHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the PyTorchLightningTrainer"""

    def __init__(self,
                 distance_metric: DistanceDefinitionSet = EuclideanDistanceDefinitionSet(),
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param runtime_mode: The device to be used
        :func:`~Trainer.__init__`
        """
        super().__init__(**kwargs)
        self.distance_metric = distance_metric


class BatchCenterLossDefinition(LossDefinition):
    """Definition of the PyTorchLightningTrainer"""

    def __init__(self, hyperparams: BatchCenterLossHyperParameterSet = BatchCenterLossHyperParameterSet()):
        super().__init__(LossType.BatchCenterLoss, hyperparams)

    def instantiate(self, *args, **kwargs):
        return BatchCenterLoss(*args, **self.hyperparams, **kwargs)


class BatchCenterLoss(nn.Module):
    """
    Batch Center loss

    2. Calculate the distances between the instances of a class
    3. Sum the distances
    """

    def __init__(self, distance_metric: DistanceDefinitionSet = EuclideanDistanceDefinitionSet()):
        super(BatchCenterLoss, self).__init__()
        self.distance_metric: Distance = distance_metric.instantiate()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        feat_size = x.size(-1)

        num_classes = target.max() + 1

        distance_sum = torch.tensor(0)
        for class_id in range(num_classes):
            sample_indices = (target == class_id).nonzero(as_tuple=False).squeeze()
            samples_in_class = x.index_select(0, sample_indices)

            distance_matrix = self.distance_metric.get_distance_matrix(samples_in_class, samples_in_class)

            distance_sum = distance_sum + distance_matrix.sum() / 2

        return distance_sum / batch_size