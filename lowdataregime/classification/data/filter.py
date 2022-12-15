"""
This module contains Dataset filter implementations
"""
from typing import List

import numpy as np
from torch.utils.data import Subset


class ClassFilter:
    """
    Class to filter a datamodule_type by label
    """

    def __init__(self, classes_to_use: List[str], num_of_images_from_sample: int = None):
        """
        Initialize a new object
        :param classes_to_use: List of labels to be used
        :param num_of_images_from_sample: The number of images which should be used for sampling
        """
        self.num_of_images_from_sample = num_of_images_from_sample
        self.classes_to_use = classes_to_use

    def filter_dataset(self, dataset):
        """
        Filters the datamodule_type according to the defined constraints
        :param dataset: The datamodule_type to be filtered
        :return: The filtered datamodule_type
        """
        if isinstance(dataset, Subset):
            main_dataset = dataset.dataset
            targets = [main_dataset.targets[idx] for idx in dataset.indices]
        else:
            main_dataset = dataset
            targets = main_dataset.targets
        class_idx = [main_dataset.class_to_idx[class_name] for class_name in self.classes_to_use]
        indices = np.where(np.isin(np.array(targets), class_idx))[0]
        if self.num_of_images_from_sample is not None:
            if self.num_of_images_from_sample > len(indices):
                raise ValueError(
                    f"The datamodule_type of the defined classes ({self.classes_to_use}) is to small ({len(indices)} images) to sample {self.num_of_images_from_sample} images!")
            indices = indices[0:self.num_of_images_from_sample]
        return Subset(dataset, indices)
