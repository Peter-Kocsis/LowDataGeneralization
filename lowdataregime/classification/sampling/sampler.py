"""
This module contains sampler implementations
"""
from abc import ABC
from typing import Callable, List

import torch
from torch.utils.data import SubsetRandomSampler

from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet, SingleDefinitionSpace, \
    SingleHyperParameterSpace
from lowdataregime.utils.utils import SerializableEnum


class SamplerType(SerializableEnum):
    """Definition of the available sampler"""
    SubsetSequentialSampler = "SubsetSequentialSampler"
    SubsetRandomSampler = "SubsetRandomSampler"

    CombineSampler = "CombineSampler"
    ClassBasedSampler = "ClassBasedSampler"

    RegionSampler = "RegionSampler"


class SamplerDefinition(DefinitionSet, ABC):
    """Abstract definition of a Sampler"""

    def __init__(self, type: SamplerType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)

    def instantiate(self, *args, **kwargs):
        return self._instantiate_func(*args, **self.hyperparams, **kwargs)


# ----------------------------------- SubsetSequentialSampler -----------------------------------


class SubsetSequentialSamplerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the SubsetSequentialSampler"""

    def __init__(self, indices: List[int] = None, **kwargs):
        """
        Creates new HyperParameterSet
        :param indices: The indices to sample from sequentially
        """
        super().__init__(**kwargs)
        self.indices = indices

    def definition_space(self):
        return SingleHyperParameterSpace(self)


class SubsetSequentialSamplerDefinition(SamplerDefinition):
    """Definition of the SubsetSequentialSampler"""

    def __init__(self,
                 hyperparams: SubsetSequentialSamplerHyperParameterSet = SubsetSequentialSamplerHyperParameterSet()):
        super().__init__(SamplerType.SubsetSequentialSampler, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return SubsetSequentialSampler

    def definition_space(self):
        return SingleDefinitionSpace(self)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Adapted from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, indices):
        """
        Creates new sampler
        :param indices: The indices to sample from sequentially
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


# ----------------------------------- SubsetRandomSampler -----------------------------------


class SubsetRandomSamplerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the SubsetRandomSampler"""

    def __init__(self, indices: List[int] = None, **kwargs):
        """
        Creates new HyperParameterSet
        :param indices: The indices to sample from randomly
        """
        super().__init__(**kwargs)
        self.indices = indices

    def definition_space(self):
        return SingleHyperParameterSpace(self)


class SubsetRandomSamplerDefinition(SamplerDefinition):
    """Definition of the SubsetRandomSampler"""

    def __init__(self, hyperparams: SubsetRandomSamplerHyperParameterSet = SubsetRandomSamplerHyperParameterSet()):
        super().__init__(SamplerType.SubsetRandomSampler, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return SubsetRandomSampler

    def definition_space(self):
        return SingleDefinitionSpace(self)
