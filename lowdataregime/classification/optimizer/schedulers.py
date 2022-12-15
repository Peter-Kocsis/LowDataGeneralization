import copy
from abc import ABC
from collections import Callable
from typing import List, Dict, Sequence, Any

import optuna
from torch.optim.lr_scheduler import MultiStepLR

from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet, DefinitionSpace, HyperParameterSpace
from lowdataregime.utils.utils import SerializableEnum


class SchedulerType(SerializableEnum):
    """Definition of the available schedulers"""
    MultiStepLR = "MultiStepLR"


class SchedulerDefinition(DefinitionSet, ABC):
    """Abstract definition of a Scheduler"""

    def __init__(self, type: SchedulerType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)

    def instantiate(self, *args, **kwargs):
        return self._instantiate_func(*args, **self.hyperparams, **kwargs)

# ----------------------------------- ADAM -----------------------------------


class MultiStepLRSchedulerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the MultiStepLRScheduler"""

    def __init__(self,
                 max_epochs: int = 200,
                 milestone_ratios: List[int] = None,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False,
                 **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~MultiStepLR.__init__`
        """
        super().__init__(**kwargs)
        self.max_epochs = max_epochs
        self.milestone_ratios = milestone_ratios
        self.milestones = None
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.verbose = verbose

    def definition_space(self):
        return MultiStepLRSchedulerHyperParameterSpace(self)


class MultiStepLRSchedulerDefinition(SchedulerDefinition):
    """Definition of the MultiStepLRScheduler"""

    def __init__(self, hyperparams: MultiStepLRSchedulerHyperParameterSet = MultiStepLRSchedulerHyperParameterSet()):
        super().__init__(SchedulerType.MultiStepLR, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        raise NotImplementedError()

    def instantiate(self, *args, **kwargs):
        self.hyperparams.milestones = [milestone_ratio * self.hyperparams.max_epochs
                                       for milestone_ratio in self.hyperparams.milestone_ratios]
        return MultiStepLR(*args,
                           milestones=self.hyperparams.milestones,
                           gamma=self.hyperparams.gamma,
                           last_epoch=self.hyperparams.last_epoch,
                           verbose=self.hyperparams.verbose,
                           **kwargs)

    def definition_space(self):
        return MultiStepLRSchedulerDefinitionSpace(self.hyperparams.definition_space())


class MultiStepLRSchedulerHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: MultiStepLRSchedulerHyperParameterSet = MultiStepLRSchedulerHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {
            "milestone_ratio": [0.15, 0.8],
        }

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise {
            "milestone_ratio": [0.15, 0.8],
        }

    def suggest(self, trial: optuna.Trial) -> MultiStepLRSchedulerHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)

        if hyperparam_set.milestone_ratios is None:
            hyperparam_set.milestone_ratios = [trial.suggest_float("milestone_ratio", 0.15, 0.8)]
        return hyperparam_set


class MultiStepLRSchedulerDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: MultiStepLRSchedulerHyperParameterSpace = MultiStepLRSchedulerHyperParameterSpace()):
        super().__init__(SchedulerType.MultiStepLR, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> MultiStepLRSchedulerDefinition:
        return MultiStepLRSchedulerDefinition(self.hyperparam_space.suggest(trial))
