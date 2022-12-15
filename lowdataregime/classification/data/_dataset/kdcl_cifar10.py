from typing import Tuple, Any

from torchvision.datasets import CIFAR10


class KDCL_CIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img1, target = super().__getitem__(index)
        img2, _ = super().__getitem__(index)
        img = (img1, img2)
        return img, target