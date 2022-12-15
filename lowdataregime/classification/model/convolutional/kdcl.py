from collections import OrderedDict
from typing import Callable, Dict, Sequence, Any

import optuna
import torch
from torch import nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet, \
    ClassificationModule, ClassificationModuleHyperParameterSpace, TrainStage
from lowdataregime.classification.model.convolutional.dml import DMLNet
from lowdataregime.classification.model.convolutional.resnet import ResNet18Definition, ResNet50Definition, \
    ResNet50HyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace
import pytorch_lightning as pl


class KDCLNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the GeneralNet"""

    def __init__(self,
                 main_net_def: ClassificationModuleDefinition = ResNet18Definition(),
                 optimizer_definition: OptimizerDefinition = None,
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param iid_net_def: The definition of the feature extraction part
        :param mpn_net_def: The definition of the message passing network part
        :func:`~ClassificationModuleHyperParameterSet.__init__`
        """
        super().__init__(optimizer_definition, scheduler_definition, loss_calc_params, **kwargs)
        self.main_model_def = main_net_def

    def definition_space(self):
        return KDCLNetHyperParameterSpace(self)


class KDCLNetDefinition(ClassificationModuleDefinition):
    """Definition of the GeneralNet"""

    def __init__(self, hyperparams: KDCLNetHyperParameterSet = KDCLNetHyperParameterSet()):
        super().__init__(ModelType.KDCLNet, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return KDCLNet

    def definition_space(self):
        return KDCLNetDefinitionSpace(self.hyperparams.definition_space())


class KDCLNet(DMLNet):
    """
    GeneralNet
    """
    def forward(self, x: torch.tensor):
        """
        Runs the forward pass on the data
        :param x: The input to be forwarded
        :return: The output of the model
        """
        x1, x2 = x
        x_main = self.model.main_model(x1)
        x_mutual = self.model.mutual_model(x2)
        return x_main, x_mutual


class KDCLNetHyperParameterSpace(ClassificationModuleHyperParameterSpace):

    def __init__(self, default_hyperparam_set: KDCLNetHyperParameterSet = KDCLNetHyperParameterSet()):
        super().__init__(default_hyperparam_set)

    def suggest(self, trial: optuna.Trial) -> KDCLNetDefinition:
        raise NotImplementedError()


class KDCLNetDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: KDCLNetHyperParameterSpace = KDCLNetHyperParameterSpace()):
        super().__init__(ModelType.GeneralNet, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> KDCLNetDefinition:
        return KDCLNetDefinition(self.hyperparam_space.suggest(trial))
