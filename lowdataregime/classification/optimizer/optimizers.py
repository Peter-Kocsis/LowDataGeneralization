import copy
import functools
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Callable, Sequence, Any

import optuna
from torch.optim import SGD, Adam

from lowdataregime.classification.optimizer.RAdam import RAdam
from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet, DefinitionSpace, HyperParameterSpace
from lowdataregime.utils.utils import SerializableEnum


class OptimizerType(SerializableEnum):
    """Definition of the available optimizers"""
    Adam = "Adam"
    RAdam = "RAdam"
    SGD = "SGD"


class OptimizerDefinition(DefinitionSet, ABC):
    """Abstract definition of a Optimizer"""

    def __init__(self, type: OptimizerType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)

    def instantiate(self, *args, **kwargs):
        return self._instantiate_func(*args, **self.hyperparams, **kwargs)


class OptimizerDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""
    pass

# ----------------------------------- ADAM -----------------------------------


class AdamOptimizerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the AdamOptimizer"""

    def __init__(self, lr: float = 3e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~Adam.__init__`
        """
        super().__init__(**kwargs)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def definition_space(self):
        return AdamOptimizerHyperParameterSpace(self)


class AdamOptimizerDefinition(OptimizerDefinition):
    """Definition of the AdamOptimizer"""

    def __init__(self, hyperparams: AdamOptimizerHyperParameterSet = AdamOptimizerHyperParameterSet()):
        super().__init__(OptimizerType.Adam, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return Adam

    def definition_space(self):
        return AdamOptimizerDefinitionSpace(self.hyperparams.definition_space())


class AdamOptimizerHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: AdamOptimizerHyperParameterSet = AdamOptimizerHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {
            "lr": [1e-2, 1e-3, 1e-4, 1e-5],
            "weight_decay": [0.0, 5e-4]
        }

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {
            "lr": [1e-5, 1e-2],
            "weight_decay": [0.0, 5e-4]
        }

    def suggest(self, trial: optuna.Trial) -> AdamOptimizerHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)

        if hyperparam_set.lr is None:
            hyperparam_set.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        if hyperparam_set.weight_decay is None:
            hyperparam_set.weight_decay = trial.suggest_float("weight_decay", 0.0, 5e-4)
        return hyperparam_set


class AdamOptimizerDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: AdamOptimizerHyperParameterSpace = AdamOptimizerHyperParameterSpace()):
        super().__init__(OptimizerType.Adam, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> AdamOptimizerDefinition:
        return AdamOptimizerDefinition(self.hyperparam_space.suggest(trial))

# ----------------------------------- RADAM -----------------------------------

class RAdamOptimizerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the AdamOptimizer"""

    def __init__(self, lr: float = 3e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~Adam.__init__`
        """
        super().__init__(**kwargs)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def definition_space(self):
        return RAdamOptimizerHyperParameterSpace(self)


class RAdamOptimizerDefinition(OptimizerDefinition):
    """Definition of the AdamOptimizer"""

    def __init__(self, hyperparams: RAdamOptimizerHyperParameterSet = RAdamOptimizerHyperParameterSet()):
        super().__init__(OptimizerType.RAdam, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return RAdam

    def definition_space(self):
        return RAdamOptimizerDefinitionSpace(self.hyperparams.definition_space())


class RAdamOptimizerHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: RAdamOptimizerHyperParameterSet = RAdamOptimizerHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {
            "lr": [1e-2, 1e-3, 1e-4, 1e-5],
            "weight_decay": [0.0, 5e-4]
        }

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {
            "lr": [1e-5, 1e-2],
            "weight_decay": [0.0, 5e-4]
        }

    def suggest(self, trial: optuna.Trial) -> RAdamOptimizerHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)

        if hyperparam_set.lr is None:
            hyperparam_set.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        if hyperparam_set.weight_decay is None:
            hyperparam_set.weight_decay = trial.suggest_float("weight_decay", 0.0, 5e-4)
        return hyperparam_set


class RAdamOptimizerDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: RAdamOptimizerHyperParameterSpace = RAdamOptimizerHyperParameterSpace()):
        super().__init__(OptimizerType.RAdam, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> RAdamOptimizerDefinition:
        return RAdamOptimizerDefinition(self.hyperparam_space.suggest(trial))

# ----------------------------------- SGD -----------------------------------


class SGDOptimizerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the SGDOptimizer"""

    def __init__(self, lr=0.1, momentum=0.9, dampening=0,
                 weight_decay=5e-4, nesterov=False, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~SGD.__init__`
        """
        super().__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def definition_space(self):
        return SGDOptimizerHyperParameterSpace(self)


class SGDOptimizerDefinition(OptimizerDefinition):
    """Definition of the SGDOptimizer"""

    def __init__(self, hyperparams: SGDOptimizerHyperParameterSet = SGDOptimizerHyperParameterSet()):
        super().__init__(OptimizerType.SGD, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return SGD

    def definition_space(self):
        return SGDOptimizerDefinitionSpace(self.hyperparams.definition_space())


class SGDOptimizerHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: SGDOptimizerHyperParameterSet = SGDOptimizerHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {
            "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            "momentum": [0.0, 0.5, 0.9],
            "weight_decay": [0.0, 5e-4]
        }

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {
            "lr": [1e-5, 1e-1],
            "momentum": [0.0, 0.9],
            "weight_decay": [0.0, 5e-4]
        }

    def suggest(self, trial: optuna.Trial) -> SGDOptimizerHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)

        if hyperparam_set.lr is None:
            hyperparam_set.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        if hyperparam_set.momentum is None:
            hyperparam_set.momentum = trial.suggest_float("momentum", 0.0, 0.9)

        if hyperparam_set.weight_decay is None:
            hyperparam_set.weight_decay = trial.suggest_float("weight_decay", 0.0, 5e-4)
        return hyperparam_set


class SGDOptimizerDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: SGDOptimizerHyperParameterSpace = SGDOptimizerHyperParameterSpace()):
        super().__init__(OptimizerType.SGD, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> SGDOptimizerDefinition:
        return SGDOptimizerDefinition(self.hyperparam_space.suggest(trial))
