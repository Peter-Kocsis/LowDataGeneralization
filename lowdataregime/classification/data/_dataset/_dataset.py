from collections import Counter
from typing import List, T

import torch
from torch import randperm
from torch.utils.data import Dataset, Subset


class DuplicateDataset(Dataset):
    def __init__(self, dataset: Dataset, duplication_factor: int):
        self.dataset = dataset
        self.duplication_factor = duplication_factor
        self.original_length = len(self.dataset)

    @property
    def targets(self):
        return self.dataset.targets

    def __getitem__(self, i):
        index_in_original_dataset = i % self.original_length
        return self.dataset[index_in_original_dataset]

    def __len__(self):
        return self.duplication_factor * self.original_length


def balanced_random_split(dataset: Dataset[T], samples_per_classes: List[int], seed: int) -> List[Subset[T]]:
    r"""
    Randomly split a balanced part from a dataset.
    Optionally fix the seed for reproducible results, e.g.:

    >>> balanced_random_split(caltech_dataset, [15, 15], seed=0)

    Arguments:
        dataset (Dataset): Dataset to be split
        samples_per_class (integer): Samples per class in the split dataset
        seed (integer): Random seed
    """
    generator = torch.Generator().manual_seed(seed)

    # Check whether split is possible
    class_counter = Counter(dataset.y)
    if min(class_counter.values()) < sum(samples_per_classes):
        raise ValueError(f"Unable to select {samples_per_classes} images! "
                         f"There is a class, which contains only {min(class_counter.values())} images!")

    indices_balanced = [[] for _ in range(len(samples_per_classes))]
    indices_rem = []

    starting_idx = 0
    for class_id in class_counter.keys():
        num_of_samples_in_class = class_counter[class_id]
        in_class_indices = (starting_idx + randperm(num_of_samples_in_class, generator=generator)).tolist()
        starting_idx += num_of_samples_in_class

        offset = 0
        for idx, count in enumerate(samples_per_classes):
            indices_balanced[idx].extend(in_class_indices[offset:offset + count])
            offset += count

        indices_rem.extend(in_class_indices[offset:])

    return [Subset(dataset, indices_balanced_batch) for indices_balanced_batch in indices_balanced] + [Subset(dataset, indices_rem)]


def balanced_random_ratio_split(dataset: Dataset[T], samples_per_classes: float, seed: int) -> List[Subset[T]]:
    r"""
    Randomly split a balanced part from a dataset.
    Optionally fix the seed for reproducible results, e.g.:

    >>> balanced_random_split(caltech_dataset, 0.7, seed=0)

    Arguments:
        dataset (Dataset): Dataset to be split
        samples_per_class (integer): Samples per class in the split dataset
        seed (integer): Random seed
    """
    generator = torch.Generator().manual_seed(seed)

    # Check whether split is possible
    class_counter = Counter(dataset.y)
    if 1.0 < samples_per_classes:
        raise ValueError(f"Unable to select {samples_per_classes} ratio of images!")

    indices_balanced = []
    indices_rem = []

    starting_idx = 0
    for class_id in class_counter.keys():
        num_of_samples_in_class = class_counter[class_id]
        in_class_indices = (starting_idx + randperm(num_of_samples_in_class, generator=generator)).tolist()
        starting_idx += num_of_samples_in_class

        count = int(samples_per_classes * num_of_samples_in_class)
        offset = 0
        indices_balanced.extend(in_class_indices[offset:offset + count])
        offset += count

        indices_rem.extend(in_class_indices[offset:])

    return [Subset(dataset, indices_balanced), Subset(dataset, indices_rem)]


def balanced_random_ratio_split(dataset: Dataset[T], samples_per_classes: float, seed: int) -> List[Subset[T]]:
    r"""
    Randomly split a balanced part from a dataset.
    Optionally fix the seed for reproducible results, e.g.:

    >>> balanced_random_split(caltech_dataset, 0.3, seed=0)

    Arguments:
        dataset (Dataset): Dataset to be split
        samples_per_class (integer): Samples per class in the split dataset
        seed (integer): Random seed
    """
    generator = torch.Generator().manual_seed(seed)

    # Check whether split is possible
    class_counter = Counter(dataset.y)
    if 1.0 < samples_per_classes:
        raise ValueError(f"Unable to select {samples_per_classes} ratio of images!")

    indices_balanced = []
    indices_rem = []

    starting_idx = 0
    for class_id in class_counter.keys():
        num_of_samples_in_class = class_counter[class_id]
        in_class_indices = (starting_idx + randperm(num_of_samples_in_class, generator=generator)).tolist()
        starting_idx += num_of_samples_in_class

        count = int(samples_per_classes * num_of_samples_in_class)
        offset = 0
        indices_balanced.extend(in_class_indices[offset:offset + count])
        offset += count

        indices_rem.extend(in_class_indices[offset:])

    return [Subset(dataset, indices_balanced), Subset(dataset, indices_rem)]
