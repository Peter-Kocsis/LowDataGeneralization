import warnings
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import Mapping
from enum import Enum
from typing import Optional, Callable, Sequence, Any, Dict

import optuna
from optuna.trial import FrozenTrial

from lowdataregime.utils.utils import Serializable, SerializableEnum


class Status(Serializable):
    def __init__(self, status_file: str = None, job_id: int = -1):
        self.status_file = status_file
        self.job_id = job_id


class HyperParameterSet(Mapping, Namespace, Serializable):
    """
    Defines all the hyper parameters required by a module
    """

    def __getitem__(self, k):
        return self.__dict__.__getitem__(k)

    def __len__(self) -> int:
        return self.__dict__.__len__()

    def __iter__(self):
        return self.__dict__.__iter__()

    def definition_space(self):
        return None


class HyperParameterSpace(ABC, Serializable):
    """
    Defines the space of hyper parameters required by a module
    """

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> HyperParameterSet:
        pass

    @property
    @abstractmethod
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        pass

    @property
    @abstractmethod
    def search_space(self) -> Dict[str, Sequence[Any]]:
        pass


class DefinitionSet(ABC, Serializable):
    """
    Defines an instance of a module
    """

    def __init__(self, type: SerializableEnum = None, hyperparams: HyperParameterSet = None):
        """
        Creates new definition
        :param type: The type of the module
        :param hyperparams: The hyperparameters of the module
        """
        self.type = type
        self.hyperparams = hyperparams

    @property
    def _instantiate_func(self) -> Optional[Callable]:  # TODO: Think about removing this method.
        """Abstract method to instantiate the module"""
        pass

    def instantiate(self, *args, **kwargs):
        """Instantiates the module"""
        return self._instantiate_func(*args, self.hyperparams, **kwargs)

    def definition_space(self):
        return None


class DefinitionSpace(ABC, Serializable):
    """
    Defines the space of module definitions
    """

    def __init__(self, type: SerializableEnum, hyperparam_space: HyperParameterSpace = None):
        self.type = type
        self.hyperparam_space = hyperparam_space

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> DefinitionSet:
        """Abstract method to suggest new definitions from the definition space"""
        pass

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return self.hyperparam_space.search_grid

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return self.hyperparam_space.search_space


class SingleHyperParameterSpace(HyperParameterSpace):
    """HyperParameterSpace with a single HyperParameterSet"""

    def __init__(self, hyperparam_set: HyperParameterSet):
        self.hyperparam_set = hyperparam_set

    def suggest(self, trial: optuna.Trial) -> HyperParameterSet:
        return self.hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {}


class SingleDefinitionSpace(DefinitionSpace):
    """DefinitionSpace with a single DefinitionSet"""

    def __init__(self, definition_set: DefinitionSet):
        super().__init__(definition_set.type)
        self.definition_set = definition_set

    def suggest(self, trial: optuna.Trial) -> DefinitionSet:
        return self.definition_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {}


class OptimizationDefinitionSet(Serializable):
    """
    Definition of an optimization
    """

    def __init__(
            self,
            data_definition: DefinitionSet = None,
            model_definition: DefinitionSet = None,
            trainer_definition: DefinitionSet = None,
            seed: Optional[int] = None):
        """
        Creates new definition
        :param data_definition: The definition of the data module
        :param model_definition: The definition of the model
        :param trainer_definition: The definition of the trainer
        :param seed: The seed used for random generation processes (NOT IMPLEMENTED YET)
        """
        self.data_definition = data_definition
        self.model_definition = model_definition
        self.trainer_definition = trainer_definition
        self.seed = seed

        if self.trainer_definition is not None and self.trainer_definition.hyperparams.fast_dev_run:
            warnings.warn(f"Setting number of data workers to 0 for fast dev run!")
            self.data_definition.hyperparams.num_workers = 0

        if self.seed is not None:
            self.trainer_definition.hyperparams.deterministic = True


    def definition_space(self):
        optimization_space = OptimizationDefinitionSpace(
            data_definition_space=self.data_definition.definition_space(),
            model_definition_space=self.model_definition.definition_space(),
            trainer_definition_space=self.trainer_definition.definition_space())

        return optimization_space


class OptimizationDefinitionSpace(Serializable):
    """
    Definition space of an optimization
    """

    def __init__(
            self,
            data_definition_space: Optional[DefinitionSpace] = None,
            model_definition_space: Optional[DefinitionSpace] = None,
            trainer_definition_space: Optional[DefinitionSpace] = None):
        """
        Creates new definition space
        :param data_definition_space: The definition space of the data module
        :param model_definition_space: The definition space of the model
        :param trainer_definition_space: The definition space of the trainer
        """
        self.data_definition_space = data_definition_space
        self.model_definition_space = model_definition_space
        self.trainer_definition_space = trainer_definition_space

    def suggest(self, trial: optuna.Trial) -> OptimizationDefinitionSet:
        return OptimizationDefinitionSet(
            data_definition=self.data_definition_space.suggest(trial),
            model_definition=self.model_definition_space.suggest(trial),
            trainer_definition=self.trainer_definition_space.suggest(trial)
        )

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        search_grid = {}
        search_grid.update(self.data_definition_space.search_grid)
        search_grid.update(self.model_definition_space.search_grid)
        search_grid.update(self.trainer_definition_space.search_grid)
        return search_grid

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        search_space = {}
        search_space.update(self.data_definition_space.search_space)
        search_space.update(self.model_definition_space.search_space)
        search_space.update(self.trainer_definition_space.search_space)
        return search_space
