"""
Implementation of the class-based sampling
Adapted from: https://github.com/dvl-tum/group_loss/
"""
import copy
from typing import Callable

from torch.utils.data.sampler import Sampler
import random

from lowdataregime.classification.sampling.sampler import SamplerDefinition, SamplerType
from lowdataregime.parameters.params import HyperParameterSet


class ClassBasedSamplerHyperParameterSet(HyperParameterSet):
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


class ClassBasedSamplerDefinition(SamplerDefinition):

    def __init__(self,
                 hyperparams: ClassBasedSamplerHyperParameterSet = ClassBasedSamplerHyperParameterSet()):
        super().__init__(SamplerType.ClassBasedSampler, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ClassBasedSampler


class ClassBasedSampler(Sampler):

    def __init__(self, indices_of_classes, num_classes_in_batch, num_samples_per_class):
        self.l_inds = indices_of_classes
        self.cl_b = num_classes_in_batch
        self.n_cl = num_samples_per_class
        self.batch_size = self.cl_b * self.n_cl
        self.flat_list = []

        self.length = sum((len(x) for x in self.l_inds.values()))
        self.max_length = max((len(x) for x in self.l_inds.values())) if len(self.l_inds) > 0 else 0

    def __iter__(self):
        # shuffle elements inside each class
        l_inds = copy.deepcopy(self.l_inds)
        for inds in l_inds.values():
            random.shuffle(inds)

        # create chunks
        class_chunks = [(key, [value[i:i + self.n_cl] for i in range(0, len(value), self.n_cl)]) for key, value in l_inds.items()]

        # add elements till every chunk has the same num of obs
        for chunks in class_chunks:
            while len(chunks[1][-1]) != self.n_cl:
                n_els = self.n_cl - len(chunks[1][-1])
                chunks[1][-1].extend(chunks[1][0][:min(n_els, len(chunks[1][-1]))])
        num_of_chunks = sum(len(x[1]) for x in class_chunks)

        # combine chunks
        batches = []
        for batch_id in range(num_of_chunks // self.cl_b):
            classes_in_batch = set()
            batch = []
            for chunk_id_in_batch in range(self.cl_b):
                class_chunks = list(sorted(class_chunks, key=lambda x: len(x[1]), reverse=True))
                max_length = len(class_chunks[0][1])
                num_of_longest_chunk_seqs = 1
                while num_of_longest_chunk_seqs < len(class_chunks) and len(class_chunks[num_of_longest_chunk_seqs][1]) == max_length:
                    num_of_longest_chunk_seqs += 1

                set_of_different_classes = set((x[0] for x in class_chunks[:num_of_longest_chunk_seqs])) - classes_in_batch
                if len(set_of_different_classes) == 0:
                    chosen_chunk = 0
                else:
                    chosen_class = random.choice(list(set_of_different_classes))
                    for idx, chunk in enumerate(class_chunks):
                        if chunk[0] == chosen_class:
                            chosen_chunk = idx
                            break
                classes_in_batch.add(class_chunks[chosen_chunk][0])
                batch.extend(class_chunks[chosen_chunk][1].pop())
            batches.append(batch)

        # shuffle the order of classes
        self.flat_list = [item for sublist in batches for item in sublist]

        return iter(self.flat_list)

    def __len__(self):
        return self.length
