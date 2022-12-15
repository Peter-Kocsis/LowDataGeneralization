from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import VisionDataset

from lowdataregime.utils.utils import SerializableEnum


class CIFAR10C_CorruptionType(SerializableEnum):
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    DEFOCUS_BLUR = "defocus_blur"
    ELASTIC_TRANSFORM = "elastic_transform"
    FOG = "fog"
    FROST = "frost"
    GAUSSIAN_BLUR = "gaussian_blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    GLASS_BLUR = "glass_blur"
    IMPULSE_NOISE = "impulse_noise"
    JPEG_COMPRESSION ="jpeg_compression"
    MOTION_BLUR = "motion_blur"
    PIXELATE = "pixelate"
    SATURATE = "saturate"
    SHOT_NOISE = "shot_noise"
    SNOW = "snow"
    SPATTER = "spatter"
    SPECKLE_NOISE = "speckle_noise"
    ZOOM_BLUR = "zoom_blur"


class CIFAR10C(VisionDataset):
    base_folder = 'cifar-10-c'
    NUM_CORRUPTION_SEVERITIES = 5
    labels_file = 'labels.npy'

    def __init__(
            self,
            root: str,
            corruption_type: CIFAR10C_CorruptionType,
            corruption_severity: int,
            train: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10C, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        assert not self.train, "CIFAR-10-C dataset can only be used for testing!"
        assert 0 <= self.corruption_severity < self.NUM_CORRUPTION_SEVERITIES, f"Undefined corruption severity {self.corruption_severity}"

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        self.data_file = os.path.join(self.root, self.base_folder, f"{corruption_type.value}.npy")
        self.targets_file = os.path.join(self.root, self.base_folder, self.labels_file)

        batch_size = 10000
        self.data = np.load(self.data_file)[self.corruption_severity * batch_size: (self.corruption_severity + 1) * batch_size]
        self.targets = np.load(self.targets_file)[self.corruption_severity * batch_size: (self.corruption_severity + 1) * batch_size].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return f"{self.corruption_type} - {self.corruption_severity}"
