import functools
import operator
from argparse import ArgumentParser
from typing import Callable, Sequence, Any, Dict

import optuna
import torch.nn as nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModule, \
    ClassificationModuleHyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace, DefinitionSpace


class DummyModuleHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the DummyModule"""

    def __init__(self,
                 hidden_dim: int = 128,
                 output_size: int = 10,
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
        self.hidden_dim = hidden_dim

        self.input_dim = (3, 32, 32)
        self.output_size = output_size

    def definition_space(self):
        return DummyModuleHyperParameterSpace(self)


class DummyModuleDefinition(ClassificationModuleDefinition):
    """Definition of the DummyModule"""

    def __init__(self, hyperparams: DummyModuleHyperParameterSet = DummyModuleHyperParameterSet()):
        super().__init__(ModelType.Dummy, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return DummyFCClassifier

    def definition_space(self):
        return DummyModuleDefinitionSpace(self.hyperparams.definition_space())


class DummyFCClassifier(ClassificationModule):
    """
    Dummy fully-connected classifier
    """

    def __init__(self, params: DummyModuleHyperParameterSet = DummyModuleHyperParameterSet()):
        super().__init__(params)

    def define_model(self):
        input_size = functools.reduce(operator.mul, list(self.params.input_dim), 1)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.Flatten(),
            nn.Linear(input_size, self.params.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.params.hidden_dim, self.params.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.params.hidden_dim, self.params.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.params.hidden_dim, self.params.output_size),
        )

    def initialize_model(self):
        pass

    @classmethod
    def add_argparse_args(cls, parent_parser):
        super_parser = super(cls, cls).add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        parser.add_argument('--input_dim', type=int)
        parser.add_argument('--hidden_dim', type=float, default=128)
        return parser


class DummyModuleHyperParameterSpace(HyperParameterSpace):
    """HyperParameterSpace of the DummyModule"""

    def __init__(self, default_hyperparam_set: DummyModuleHyperParameterSet = DummyModuleHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set
        self.optimizer_space = default_hyperparam_set.optimizer_definition.definition_space() \
            if default_hyperparam_set.optimizer_definition is not None else None
        self.scheduler_space = default_hyperparam_set.scheduler_definition.definition_space() \
            if default_hyperparam_set.scheduler_definition is not None else None

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        search_grid = {"hidden_dim": [32, 64, 128, 256]}
        if self.optimizer_space is not None:
            search_grid.update(self.optimizer_space.search_grid)

        if self.scheduler_space is not None:
            search_grid.update(self.scheduler_space.search_grid)
        return search_grid

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> HyperParameterSet:
        hyperparam_set = self.default_hyperparam_set
        hyperparam_set.hidden_dim = trial.suggest_int("hidden_dim", 32, 256)

        hyperparam_set.optimizer_definition = self._suggest_optimizer_definition(trial)
        hyperparam_set.scheduler_definition = self._suggest_scheduler_definition(trial)

        return hyperparam_set

    def _suggest_optimizer_definition(self, trial: optuna.Trial):
        if self.optimizer_space is None:
            return None
        return self.optimizer_space.suggest(trial)

    def _suggest_scheduler_definition(self, trial: optuna.Trial):
        if self.scheduler_space is None:
            return None
        return self.scheduler_space.suggest(trial)


class DummyModuleDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: DummyModuleHyperParameterSpace = DummyModuleHyperParameterSpace()):
        super().__init__(ModelType.Dummy, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> DummyModuleDefinition:
        return DummyModuleDefinition(self.hyperparam_space.suggest(trial))


