import copy
import functools
import random
from collections import defaultdict, Counter

import numpy as np
from typing import Optional, List, Any, Callable

from torch.utils.data import DataLoader

from lowdataregime.classification.data.cifar import CIFARDataModule
from lowdataregime.classification.sampling.sampler import SamplerType, SubsetRandomSamplerDefinition, SamplerDefinition, \
    SubsetRandomSamplerHyperParameterSet
from lowdataregime.classification.log.logger import init_logger
from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet, Status
from lowdataregime.utils.utils import pl_worker_init_function


class ActiveDataModuleHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 initial_pool_seed: int = 0,
                 train_sampler_definition: SamplerDefinition = SubsetRandomSamplerDefinition(),
                 test_sampler_definition: SamplerDefinition = SubsetRandomSamplerDefinition(),
                 use_validation: bool = False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.initial_pool_seed = initial_pool_seed
        self.train_sampler_definition = train_sampler_definition
        self.test_sampler_definition = test_sampler_definition
        self.use_validation = use_validation


class ActiveDataModuleStatus(Status):
    def __init__(self,
                 status_file: str = None,
                 job_id: int = -1,
                 labeled_pool_indices: List[int] = None,
                 unlabeled_pool_indices: List[int] = None,
                 validation_pool_indices: List[int] = None,
                 sample_occurences_in_labeled_pool: dict = None):
        super().__init__(status_file, job_id)
        self.labeled_pool_indices = labeled_pool_indices
        self.unlabeled_pool_indices = unlabeled_pool_indices
        self.validation_pool_indices = validation_pool_indices
        self.sample_occurences_in_labeled_pool = sample_occurences_in_labeled_pool


class ActiveDataModuleDefinition(DefinitionSet):
    """Definition of the ActiveTrainer"""

    def __init__(self,
                 hyperparams: ActiveDataModuleHyperParameterSet = ActiveDataModuleHyperParameterSet(),
                 status: Optional[ActiveDataModuleStatus] = None):
        super().__init__(None, hyperparams)
        self.status = status

    @property
    def _instantiate_func(self) -> Callable:
        # TODO: Make it cleaner -> maybe remove
        return None

    def instantiate(self, data_definition, *args, **kwargs):
        """Instantiates the module"""
        return ActiveDataModule(data_definition, self.hyperparams, self.status)


class ActiveDataModule:
    def __init__(self,
                 data_definition: DefinitionSet,
                 params: ActiveDataModuleHyperParameterSet = ActiveDataModuleHyperParameterSet(),
                 status: Optional[ActiveDataModuleStatus] = None,):
        self.params = params
        self.status = self._get_status(status)
        self.logger = init_logger(self.__class__.__name__)

        self.dataset: CIFARDataModule = data_definition.instantiate()

        self._unaugmented_dataset = None
        self._current_seed = self.params.initial_pool_seed

    def _get_status(self, status: Optional[ActiveDataModuleStatus] = None) -> ActiveDataModuleStatus:
        if status is None:
            return ActiveDataModuleStatus(None)
        else:
            return status

    @property
    def labeled_pool_indices(self):
        if self.status.labeled_pool_indices is None:
            self.status.labeled_pool_indices = []
            self.update_sample_occurences()
        return self.status.labeled_pool_indices

    @property
    def labeled_pool_indices_map_by_class(self):
        if self.status.labeled_pool_indices is None:
            return dict(list())
        else:
            list_of_indices_for_each_class = defaultdict(list)
            for idx in self.status.labeled_pool_indices:
                _, target = self.dataset.dataset_train[idx]
                list_of_indices_for_each_class[target].append(idx)
            return list_of_indices_for_each_class

    @property
    def labeled_pool_size(self):
        return len(self.labeled_pool_indices)

    def calculate_sample_occurences_in_labeled_pool(self, index_pool):
        max_occurence = self.dataset.duplication_factor

        original_indices = (idx % self.dataset.original_train_len for idx in index_pool)
        sample_counter = Counter(original_indices)
        freq_counter = Counter(sample_counter.values())
        return {occurence: freq_counter[occurence] for occurence in range(1, max_occurence + 1)}

    def update_sample_occurences(self):
        self.status.sample_occurences_in_labeled_pool = \
            self.calculate_sample_occurences_in_labeled_pool(self.status.labeled_pool_indices)

    @property
    def unlabeled_pool_indices(self):
        if self.status.unlabeled_pool_indices is None:
            self.status.unlabeled_pool_indices = list(set(range(self.dataset.train_len)) - set(self.labeled_pool_indices))
        return self.status.unlabeled_pool_indices

    @unlabeled_pool_indices.setter
    def unlabeled_pool_indices(self, val):
        self.status.unlabeled_pool_indices = val

    @property
    def validation_pool_indices(self):
        if self.status.validation_pool_indices is None:
            self.status.validation_pool_indices = []
        return self.status.validation_pool_indices

    @validation_pool_indices.setter
    def validation_pool_indices(self, val):
        self.status.validation_pool_indices = val

    def _is_consistent(self):
        labeled_pool_set = set(self.labeled_pool_indices)
        unlabeled_pool_set = set(self.unlabeled_pool_indices)
        validation_pool_set = set(self.validation_pool_indices)
        return len(labeled_pool_set.intersection(unlabeled_pool_set)) == 0 \
               and len(labeled_pool_set.intersection(validation_pool_set)) == 0 \
               and len(unlabeled_pool_set.intersection(validation_pool_set)) == 0 \
               and len(self.labeled_pool_indices) + len(self.unlabeled_pool_indices) + len(self.validation_pool_indices) == self.dataset.train_len

    @property
    def unaugmented_dataset_train(self):
        if self._unaugmented_dataset is None:
            self._unaugmented_dataset = self.dataset.unaugmented_dataset_train
        return self._unaugmented_dataset

    @property
    def batch_size(self):
        return self.dataset.batch_size

    @property
    def dims(self):
        return self.dataset.dims

    @property
    def num_workers(self):
        return self.dataset.num_workers

    def sort_unlabeled_indices_randomly(self, seed=None):
        """
        Sorts the unlabeled pool randomly
        """
        if seed is None:
            seed = self._current_seed
            self._current_seed += 1
        self.logger.info(f"Sorting the unlabelled pool randomly - seed: {seed}!")
        random.Random(seed).shuffle(self.unlabeled_pool_indices)
        self.logger.info("Unlabelled pool sorted randomly!")

    def sort_unlabeled_indices_by_list(self, keys):
        len_keys = len(keys)

        sorted_key_idxs = np.flip(np.argsort(keys)).tolist()
        self.unlabeled_pool_indices = \
            [self.unlabeled_pool_indices[sorted_key_idx] for sorted_key_idx in sorted_key_idxs] \
            + self.unlabeled_pool_indices[len_keys:]

    def label_initial_samples(self, num_of_samples_to_label, feature_model=None, device=None):
        self.sort_unlabeled_indices_randomly(self.params.initial_pool_seed)
        self.label_samples(num_of_samples_to_label, feature_model, device)

    def prepare_validation_set(self):
        if self.params.use_validation:
            num_of_samples_of_validation = self.dataset.test_len
            self.sort_unlabeled_indices_randomly(self.params.initial_pool_seed)
            self.validation_pool_indices = self.unlabeled_pool_indices[-num_of_samples_of_validation:]
            self.unlabeled_pool_indices = self.unlabeled_pool_indices[:-num_of_samples_of_validation]

            self.dataset.dataset_valid = copy.deepcopy(self.dataset.dataset_train)
            self.dataset.val_sampler = SubsetRandomSamplerDefinition(
                SubsetRandomSamplerHyperParameterSet(
                    indices=self.validation_pool_indices
                )).instantiate()

    def label_samples(self, num_of_samples_to_label, feature_model=None, device=None):
        """
        Labels the first <num_of_samples_to_label> elements of the unlabeled pool
        :param datamodule: The used datamodule, where the labeling takes place
        :param num_of_samples_to_label: The number unlabeled samples to be labeled
        """
        self.logger.info(
            f"Labelling first {num_of_samples_to_label} samples from the unlabelled pool!")
        if num_of_samples_to_label > len(self.unlabeled_pool_indices):
            raise ValueError(
                f"Unable to label {num_of_samples_to_label} from {len(self.unlabeled_pool_indices)} unlabeled samples")

        self.labeled_pool_indices.extend(self.unlabeled_pool_indices[:num_of_samples_to_label])
        self.unlabeled_pool_indices = self.unlabeled_pool_indices[num_of_samples_to_label:]

        self.update_sample_occurences()

        if not self._is_consistent():
            raise ValueError("Labeled and unlabeled indices are inconsistent")

        self.update_train_sampler(feature_model, device)
        self.update_test_sampler(feature_model, device)
        self.logger.info(
            f"First {num_of_samples_to_label} samples from the unlabelled pool labelled!")

    def train_labeled_sampler(self, feature_model=None, device=None):
        # Train sampler
        sampler_definition = copy.deepcopy(self.params.train_sampler_definition)
        if sampler_definition.type == SamplerType.SubsetRandomSampler:
            sampler_definition.hyperparams.indices = self.labeled_pool_indices
        elif sampler_definition.type == SamplerType.CombineSampler:
            sampler_definition.hyperparams.indices_of_classes = self.labeled_pool_indices_map_by_class
            assert sampler_definition.hyperparams.num_classes_in_batch * \
                   sampler_definition.hyperparams.num_samples_per_class == self.dataset.batch_size
        elif sampler_definition.type == SamplerType.ClassBasedSampler:
            sampler_definition.hyperparams.indices_of_classes = self.labeled_pool_indices_map_by_class
            assert sampler_definition.hyperparams.num_classes_in_batch * \
                   sampler_definition.hyperparams.num_samples_per_class == self.dataset.batch_size
        elif sampler_definition.type == SamplerType.RegionSampler:
            sampler_definition.hyperparams.indices = self.labeled_pool_indices
        else:
            raise NotImplementedError()

        sampler = sampler_definition.instantiate()

        if sampler_definition.type == SamplerType.RegionSampler:
            sampler.feature_model = feature_model
            sampler.device = device
            sampler.dataset = self.unaugmented_dataset_train

        return sampler

    def train_unlabeled_sampler(self, feature_model=None, device=None):
        # Train sampler
        sampler_definition = copy.deepcopy(self.params.train_sampler_definition)
        if sampler_definition.type == SamplerType.SubsetRandomSampler:
            sampler_definition.hyperparams.indices = self.unlabeled_pool_indices
        elif sampler_definition.type == SamplerType.CombineSampler:
            raise NotImplementedError("CombineSampler for unlabeled pool requires pseudo labeling")
        elif sampler_definition.type == SamplerType.ClassBasedSampler:
            raise NotImplementedError("ClassBasedSampler for unlabeled pool requires pseudo labeling")
        elif sampler_definition.type == SamplerType.RegionSampler:
            sampler_definition.hyperparams.indices = self.unlabeled_pool_indices
        else:
            raise NotImplementedError()

        sampler = sampler_definition.instantiate()

        if sampler_definition.type == SamplerType.RegionSampler:
            sampler.feature_model = feature_model
            sampler.device = device
            sampler.dataset = self.unaugmented_dataset_train

        return sampler

    def update_train_sampler(self, feature_model=None, device=None):
        # Train sampler
        self.dataset.train_sampler = self.train_labeled_sampler(feature_model=feature_model, device=device)

    def update_test_sampler(self, feature_model=None, device=None):
        test_indices = list(range(self.dataset.test_len))

        # Test sampler
        if self.params.test_sampler_definition.type == SamplerType.SubsetRandomSampler:
            self.params.test_sampler_definition.hyperparams.indices = test_indices
        elif self.params.test_sampler_definition.type == SamplerType.CombineSampler:
            raise NotImplementedError("CombineSampler for testing requires pseudo labeling")
        elif self.params.test_sampler_definition.type == SamplerType.ClassBasedSampler:
            raise NotImplementedError("ClassBasedSampler for testing requires pseudo labeling")
        elif self.params.test_sampler_definition.type == SamplerType.RegionSampler:
            self.params.test_sampler_definition.hyperparams.indices = test_indices
        else:
            raise NotImplementedError()

        self.dataset.test_sampler = self.params.test_sampler_definition.instantiate()

        if self.params.test_sampler_definition.type == SamplerType.RegionSampler:
            self.dataset.test_sampler.feature_model = feature_model
            self.dataset.test_sampler.device = device
            self.dataset.test_sampler.dataset = self.dataset.dataset_test

    def train_labeled_dataloader(self):
        self.logger.debug(f"Creating train dataloader!")
        loader = DataLoader(
            self.dataset.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=self.train_labeled_sampler(),
            worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
        )
        self.logger.debug(
            f"Train labeled dataloader created with dataset length: {len(loader.dataset)}, sampler length: {len(loader.sampler) if loader.sampler is not None else None}!")
        return loader

    def train_unlabeled_dataloader(self, dataset = None, batch_size: int = None):
        if dataset is None:
            dataset = self.dataset.dataset_train
        if batch_size is None:
            batch_size = self.batch_size

        self.logger.debug(f"Creating train dataloader!")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=self.train_unlabeled_sampler(),
            worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
        )
        self.logger.debug(
            f"Train unlabeled dataloader created with dataset length: {len(loader.dataset)}, sampler length: {len(loader.sampler) if loader.sampler is not None else None}!")
        return loader

    def test_dataloader(self):
        return self.dataset.test_dataloader()

