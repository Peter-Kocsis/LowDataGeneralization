"""
Implementation of the class-based sampling
Adapted from: https://github.com/dvl-tum/group_loss/
"""
from typing import Callable

from torch.utils.data.sampler import Sampler
import random

from lowdataregime.classification.sampling.sampler import SamplerDefinition, SamplerType
from lowdataregime.parameters.params import HyperParameterSet


class CombineSamplerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the SubsetSequentialSampler"""

    def __init__(self,
                 indices_of_classes = None,
                 num_classes_in_batch = None,
                 num_samples_per_class = None,
                 **kwargs):
        """
        Creates new HyperParameterSet
        """
        super().__init__(**kwargs)
        self.indices_of_classes = indices_of_classes
        self.num_classes_in_batch = num_classes_in_batch
        self.num_samples_per_class = num_samples_per_class


class CombineSamplerDefinition(SamplerDefinition):

    def __init__(self,
                 hyperparams: CombineSamplerHyperParameterSet = CombineSamplerHyperParameterSet()):
        super().__init__(SamplerType.CombineSampler, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return CombineSampler


class CombineSampler(Sampler):

    def __init__(self, indices_of_classes, num_classes_in_batch, num_samples_per_class):
        self.l_inds = indices_of_classes
        self.max = -1
        self.cl_b = num_classes_in_batch
        self.n_cl = num_samples_per_class
        self.batch_size = self.cl_b * self.n_cl
        self.flat_list = []
        self.length = sum((len(x) for x in self.l_inds.values()))

        for inds in self.l_inds.values():
            if len(inds) > self.max:
                self.max = len(inds)

    def __iter__(self):
        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds.values()))

        # add elements till every class has the same num of obs
        for inds in l_inds:
            while len(inds) != self.max:
                n_els = self.max - len(inds)
                inds.extend(inds[:min(n_els, len(inds))])  # max + 1

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            # drop the last < n_cl elements
            while len(inds) >= self.n_cl:
                split_list_of_indices.append(inds[:self.n_cl])
                inds = inds[self.n_cl:]

        # shuffle the order of classes
        random.shuffle(split_list_of_indices)
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]

        return iter(self.flat_list)

    def __len__(self):
        return self.length
