"""
Adapted from https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
"""
from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor

from lowdataregime.classification.loss.losses import LossDefinition, LossType
from lowdataregime.parameters.params import HyperParameterSet


class CenterLossHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the PyTorchLightningTrainer"""

    def __init__(self,
                 num_classes: int = 10,
                 feat_dim: int = 2,
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param runtime_mode: The device to be used
        :func:`~Trainer.__init__`
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.feat_dim = feat_dim


class CenterLossDefinition(LossDefinition):
    """Definition of the PyTorchLightningTrainer"""

    def __init__(self, hyperparams: CenterLossHyperParameterSet = CenterLossHyperParameterSet()):
        super().__init__(LossType.CenterLoss, hyperparams)

    def instantiate(self, *args, **kwargs):
        return CenterLoss(*args, **self.hyperparams, **kwargs)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes: int = 10, feat_dim: int = 2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, target):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # (a-b)^2 = a^2 + b^2 - 2*a*b
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes, dtype=torch.int64, device=x.device)

        labels = target.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss