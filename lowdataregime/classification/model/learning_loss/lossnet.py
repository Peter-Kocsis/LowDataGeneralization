import functools
import operator
from argparse import ArgumentParser
from typing import Callable, Sequence, Any, Dict, List

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModule, \
    ClassificationModuleHyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace, DefinitionSpace


class LossNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the LossNet"""

    def __init__(self,
                 feature_sizes: List[int] = [32, 16, 8, 4],
                 num_channels: List[int] = [64, 128, 256, 512],
                 interm_dim: int = 128,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param hidden_dim: The hidden dimension of the fully connected network
        :param output_size: The size of the output
        :func:`~ClassificationModuleHyperParameterSet.__init__`
        """
        super().__init__(optimizer_definition, scheduler_definition, loss_calc_params, **kwargs)
        self.feature_sizes = feature_sizes
        self.num_channels = num_channels
        self.interm_dim = interm_dim

    def definition_space(self):
        return LossNetHyperParameterSpace(self)


class LossNetDefinition(ClassificationModuleDefinition):
    """Definition of the LossNet"""

    def __init__(self, hyperparams: LossNetHyperParameterSet = LossNetHyperParameterSet()):
        super().__init__(ModelType.Dummy, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return LossNet

    def definition_space(self):
        return LossNetDefinitionSpace(self.hyperparams.definition_space())


class LossNet(ClassificationModule):
    """
    Dummy fully-connected classifier
    """

    def __init__(self, params: LossNetHyperParameterSet = LossNetHyperParameterSet()):
        super().__init__(params)

    def define_model(self):
        feature_sizes = self.params.feature_sizes
        num_channels = self.params.num_channels
        interm_dim = self.params.interm_dim

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

        return None

    def forward(self, features, *args, **kwargs):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

    @property
    def feature_model(self):
        return None

    def initialize_model(self):
        pass

    @classmethod
    def add_argparse_args(cls, parent_parser):
        return parent_parser


class LossNetHyperParameterSpace(HyperParameterSpace):
    """HyperParameterSpace of the LossNet"""

    def __init__(self, default_hyperparam_set: LossNetHyperParameterSet = LossNetHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> HyperParameterSet:
        raise NotImplementedError()


class LossNetDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: LossNetHyperParameterSpace = LossNetHyperParameterSpace()):
        super().__init__(ModelType.Dummy, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> LossNetDefinition:
        return LossNetDefinition(self.hyperparam_space.suggest(trial))


