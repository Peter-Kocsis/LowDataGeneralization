import numpy as np
from typing import Optional, Callable

from torchvision.datasets import CIFAR10

from lowdataregime.active_learning.active_datamodule import ActiveDataModuleStatus


class CIFAR10_SPLIT(CIFAR10):
    def __init__(self,
                 root: str,
                 split_def: ActiveDataModuleStatus = None,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

        if split_def is not None:
            self.data = self.data[split_def.labeled_pool_indices]
            self.targets = np.array(self.targets)[split_def.labeled_pool_indices]
