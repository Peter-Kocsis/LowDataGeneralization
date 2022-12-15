"""
Implementation of the region sampling
"""
from itertools import islice
from typing import Callable, List

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import random

from lowdataregime.classification.log.logger import init_logger
from lowdataregime.classification.sampling.sampler import SamplerDefinition, SamplerType, SubsetSequentialSampler
from lowdataregime.parameters.params import HyperParameterSet
from lowdataregime.utils.distance import DistanceDefinitionSet, CosineDistanceDefinitionSet
from lowdataregime.utils.utils import take


class RegionSamplerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the SubsetSequentialSampler"""

    def __init__(self,
                 indices=None,
                 region_size: int = None,
                 num_data_workers: int = 4,
                 distance_metric: DistanceDefinitionSet = CosineDistanceDefinitionSet(),
                 **kwargs):
        """
        Creates new HyperParameterSet
        """
        super().__init__(**kwargs)
        self.indices = indices
        self.region_size = region_size
        self.num_data_workers = num_data_workers
        self.distance_metric = distance_metric


class RegionSamplerDefinition(SamplerDefinition):

    def __init__(self,
                 hyperparams: RegionSamplerHyperParameterSet = RegionSamplerHyperParameterSet()):
        super().__init__(SamplerType.RegionSampler, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return RegionSampler


class RegionSampler(Sampler):

    def __init__(self,
                 feature_model=None,
                 device=None,
                 dataset=None,
                 indices: List[int] = None,
                 region_size: int = None,
                 num_data_workers: int = 4,
                 distance_metric: DistanceDefinitionSet = DistanceDefinitionSet()):

        self.feature_model = feature_model
        self.device = device
        self.dataset = dataset
        self.indices = indices
        self.region_size = region_size
        self.num_data_workers = num_data_workers
        self.distance_metric = distance_metric.instantiate()

        self.module_logger = init_logger("RegionSampler")

    def __iter__(self):
        assert self.feature_model is not None, "Feature model is not defined for RegionSampler!"
        assert self.dataset is not None, "Datamodule is not defined for RegionSampler!"

        random.shuffle(self.indices)

        # Step 1: Evaluate the features
        features = self._get_features()

        # Step 2: Calculate distances and nearest samples
        distances = self.distance_metric.get_distance_matrix(features, features)
        nearest_samples = torch.argsort(distances, dim=1).detach().cpu().numpy()

        # Step 3: Iterate until we have enough samples for a batch
        idx_pool = set(range(len(self.indices)))
        while len(idx_pool) >= self.region_size:
            # Step 4: Select a random sample
            idx = random.sample(idx_pool, 1)[0]
            yield self.indices[idx]
            idx_pool.remove(idx)

            # Step 6: Select nearest samples
            for next_nearest_sample_id in islice(filter(lambda x: x in idx_pool, nearest_samples[idx]), self.region_size - 1):
                yield self.indices[next_nearest_sample_id]
                idx_pool.remove(next_nearest_sample_id)

    def _get_features(self):
        loader = DataLoader(self.dataset,
                            batch_size=100,
                            sampler=SubsetSequentialSampler(torch.tensor(self.indices)),
                            num_workers=self.num_data_workers,
                            drop_last=True,
                            # more convenient if we maintain the order of subset
                            pin_memory=True)

        features = torch.tensor([], device=self.device)

        with torch.no_grad():
            for (inputs, labels) in loader:
                inputs = inputs.to(self.device)
                batch_features, _ = self.feature_model(inputs)
                features = torch.cat((features, batch_features), 0)

        return features

    def __len__(self):
        return len(self.indices)
