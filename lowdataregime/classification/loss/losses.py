from abc import ABC
from typing import Optional, Any, Sequence, Dict

import optuna
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable

from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet, DefinitionSpace, HyperParameterSpace
from lowdataregime.utils.utils import SerializableEnum


class LossType(SerializableEnum):
    """Definition of the available losses"""
    CrossEntropy = "CrossEntropy"
    CenterLoss = "CenterLoss"
    BatchCenterLoss = "BatchCenterLoss"
    DMLLoss = "DMLLoss"
    KDCLLoss = "KDCLLoss"
    KDLoss = "KDLoss"


class LossDefinition(DefinitionSet, ABC):
    """Abstract definition of a Loss"""

    def __init__(self, type: LossType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)


# ----------------------------------- CrossEntropy -----------------------------------


class CrossEntropyHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the PyTorchLightningTrainer"""

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 ignore_index: int = -100,
                 reduction: str = 'mean',
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param runtime_mode: The device to be used
        :func:`~Trainer.__init__`
        """
        super().__init__(**kwargs)

        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction


class CrossEntropyDefinition(LossDefinition):
    """Definition of the PyTorchLightningTrainer"""

    def __init__(self, hyperparams: CrossEntropyHyperParameterSet = CrossEntropyHyperParameterSet()):
        super().__init__(LossType.CrossEntropy, hyperparams)

    def instantiate(self, *args, **kwargs):
        return nn.CrossEntropyLoss(*args, **self.hyperparams, **kwargs)


# ----------------------------------- DML Loss -----------------------------------


class DMLLossHyperParameterSet(HyperParameterSet):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class DMLLossDefinition(LossDefinition):
    def __init__(self, hyperparams: DMLLossHyperParameterSet = DMLLossHyperParameterSet()):
        super().__init__(LossType.DMLLoss, hyperparams)

    def instantiate(self, *args, **kwargs):
        return DMLLoss(*args, **kwargs)


class DMLLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, input: Tensor, target: Tensor):
        pred_main, pred_mutual = input
        # Calculate CE for both models
        ce_loss = self.loss_ce(pred_main, target) + self.loss_ce(pred_mutual, target)

        # Calculate KL for both directions, but detach the other model always
        main_log_softmax = F.log_softmax(pred_main, dim=1)
        mutual_log_softmax = F.log_softmax(pred_mutual, dim=1)

        main_softmax = F.softmax(pred_main, dim=1)
        mutual_softmax = F.softmax(pred_mutual, dim=1)

        kl_loss = self.loss_kl(main_log_softmax, mutual_softmax.detach()) + self.loss_kl(mutual_log_softmax, main_softmax.detach())

        return ce_loss + kl_loss


# ----------------------------------- KDCL Loss -----------------------------------


class KDCLLossHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 temperature: int = 2,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.temperature = temperature


class KDCLLossDefinition(LossDefinition):
    def __init__(self, hyperparams: KDCLLossHyperParameterSet = KDCLLossHyperParameterSet()):
        super().__init__(LossType.KDCLLoss, hyperparams)

    def instantiate(self, *args, **kwargs):
        return KDCLLoss(self.hyperparams, *args, **kwargs)


class KDCLLoss(nn.Module):
    def __init__(self, params: KDCLLossHyperParameterSet = None):
        super().__init__()
        self.params = params
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, input: Tensor, target: Tensor):
        pred_main, pred_mutual = input

        # Calculate the soft target
        soft_target = (pred_main + pred_mutual) / 2
        soft_target = soft_target.detach()

        # Calculate CE for both models
        ce_loss = self.loss_ce(pred_main, target) + self.loss_ce(pred_mutual, target)

        # Calculate KL for both directions, but detach the other model always
        kl_loss = self.loss_kl(F.log_softmax(pred_main / self.params.temperature, dim=1), F.softmax(soft_target / self.params.temperature, dim=1)) * self.params.temperature * self.params.temperature + \
                  self.loss_kl(F.log_softmax(pred_mutual / self.params.temperature, dim=1), F.softmax(soft_target / self.params.temperature, dim=1)) * self.params.temperature * self.params.temperature

        return ce_loss + kl_loss


# ----------------------------------- KD Loss -----------------------------------


class KDLossHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 temperature: int = 2,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.temperature = temperature


class KDLossDefinition(LossDefinition):
    def __init__(self, hyperparams: KDLossHyperParameterSet = KDLossHyperParameterSet()):
        super().__init__(LossType.KDLoss, hyperparams)

    def instantiate(self, *args, **kwargs):
        return KDLoss(self.hyperparams, *args, **kwargs)


class KDLoss(nn.Module):
    def __init__(self, params: KDCLLossHyperParameterSet = None):
        super().__init__()
        self.params = params
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, input: Tensor, target: Tensor):
        pred_student, pred_teacher = input

        # Calculate the soft target
        soft_target = pred_teacher.detach()

        # Calculate CE for both models
        ce_loss = self.loss_ce(pred_student, target)

        # Calculate KL for both directions, but detach the other model always
        kl_loss = self.loss_kl(F.log_softmax(pred_student / self.params.temperature, dim=1), F.softmax(soft_target / self.params.temperature, dim=1)) * self.params.temperature * self.params.temperature

        return ce_loss + kl_loss
