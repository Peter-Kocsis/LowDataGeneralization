from abc import ABC

from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet
from lowdataregime.utils.utils import SerializableEnum


class DataModuleType(SerializableEnum):
    """Definition of the implemented data modules"""
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    CALTECH101 = "caltech101"
    CALTECH256 = "caltech256"


class DataModuleDefinition(DefinitionSet, ABC):
    """Abstract definition of a DataModule"""

    def __init__(self, type: DataModuleType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)
