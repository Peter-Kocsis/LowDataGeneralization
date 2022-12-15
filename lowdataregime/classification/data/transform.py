from abc import ABC
from enum import Enum
from typing import Dict, Callable, List

import torch
import torchvision.transforms as T
from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet
from lowdataregime.utils.utils import SerializableEnum


class TransformType(SerializableEnum):
    """Definition of the available transforms"""
    RandomHorizontalFlip = "RandomHorizontalFlip"
    RandomCrop = "RandomCrop"
    ToTensor = "ToTensor"
    Normalize = "Normalize"
    Resize = "Resize"
    Repeat = "Repeat"
    Compose = "Compose"


class TransformDefinition(DefinitionSet, ABC):
    """Abstract definition of a Transform"""

    def __init__(self, type: TransformType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)

    def instantiate(self, *args, **kwargs):
        return self._instantiate_func(*args, **self.hyperparams, **kwargs)


# ----------------------------------- RandomHorizontalFlip -----------------------------------


class RandomHorizontalFlipTransformHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the RandomHorizontalFlipTransform"""

    def __init__(self, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~T.RandomHorizontalFlip.__init__`
        """
        super().__init__(**kwargs)


class RandomHorizontalFlipTransformDefinition(TransformDefinition):
    """Definition of the RandomHorizontalFlipTransform"""

    def __init__(self,
                 hyperparams: RandomHorizontalFlipTransformHyperParameterSet = RandomHorizontalFlipTransformHyperParameterSet()):
        super().__init__(TransformType.RandomHorizontalFlip, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return T.RandomHorizontalFlip


# ----------------------------------- RandomCrop -----------------------------------


class RandomCropTransformHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the RandomCropTransform"""

    def __init__(self, size=32, padding=4, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~T.RandomCrop.__init__`
        :param size: The size of the cropped region
        :param padding: Padding to be added
        """
        super().__init__(**kwargs)
        self.size = size
        self.padding = padding


class RandomCropTransformDefinition(TransformDefinition):
    """Definition of the RandomCropTransform"""

    def __init__(self, hyperparams: RandomCropTransformHyperParameterSet = RandomCropTransformHyperParameterSet()):
        super().__init__(TransformType.RandomCrop, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return T.RandomCrop


# ----------------------------------- ToTensor -----------------------------------


class ToTensorTransformHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the ToTensorTransform"""

    def __init__(self, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~T.ToTensor.__init__`
        """
        super().__init__(**kwargs)


class ToTensorTransformDefinition(TransformDefinition):
    """Definition of the ToTensorTransform"""

    def __init__(self, hyperparams: ToTensorTransformHyperParameterSet = ToTensorTransformHyperParameterSet()):
        super().__init__(TransformType.ToTensor, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return T.ToTensor


# ----------------------------------- Normalize -----------------------------------


class NormalizeHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the NormalizeTransform"""

    def __init__(self, mean=None, std=None, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~T.Normalize.__init__`
        :param mean: The mean to normalize to
        :param std: The standard deviation to normalize to
        """
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std


class NormalizeTransformDefinition(TransformDefinition):
    """Definition of the NormalizeTransform"""

    def __init__(self, hyperparams: NormalizeHyperParameterSet = NormalizeHyperParameterSet()):
        super().__init__(TransformType.Normalize, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return T.Normalize


# ----------------------------------- Resize -----------------------------------


class ResizeHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the NormalizeTransform"""

    def __init__(self, size=None, **kwargs):
        """
        Creates new HyperParameterSet
        :func:`~T.Resize.__init__`
        :param mean: The mean to normalize to
        :param std: The standard deviation to normalize to
        """
        super().__init__(**kwargs)
        self.size = size


class ResizeTransformDefinition(TransformDefinition):
    """Definition of the NormalizeTransform"""

    def __init__(self, hyperparams: ResizeHyperParameterSet = ResizeHyperParameterSet()):
        super().__init__(TransformType.Resize, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return T.Resize


# ----------------------------------- Repeat -----------------------------------


class Repeat(torch.nn.Module):
    def __init__(self, desired_num_of_channels):
        super().__init__()
        self.desired_num_of_channels = desired_num_of_channels

    def forward(self, img):
        if img.ndim == 4:
            repeat_required = self.desired_num_of_channels // img.shape[1]
            return img.repeat(1, repeat_required, 1, 1)
        elif img.ndim == 3:
            repeat_required = self.desired_num_of_channels // img.shape[0]
            return img.repeat(repeat_required, 1, 1)
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__ + '(desired_num_of_channels={0})'.format(self.desired_num_of_channels)


class RepeatHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the RepeatHyperParameterSet"""

    def __init__(self, desired_num_of_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.desired_num_of_channels = desired_num_of_channels


class RepeatTransformDefinition(TransformDefinition):
    """Definition of the NormalizeTransform"""

    def __init__(self, hyperparams: RepeatHyperParameterSet = RepeatHyperParameterSet()):
        super().__init__(TransformType.Repeat, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return Repeat


# ----------------------------------- Compose -----------------------------------


class ComposeTransformHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the ComposeTransform"""

    def __init__(self, transforms: List[TransformDefinition] = None, **kwargs):
        """
        Creates new HyperParameterSet
        :param transforms: List of TransformDefinitions to be composed
        """
        super().__init__(**kwargs)
        self.transforms = transforms


class ComposeTransformDefinition(TransformDefinition):
    """Definition of the ComposeTransform"""

    def __init__(self, hyperparams: ComposeTransformHyperParameterSet = ComposeTransformHyperParameterSet()):
        super().__init__(TransformType.Compose, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return T.Compose

    def instantiate(self, *args, **kwargs):
        transforms = [transform_def.instantiate() for transform_def in self.hyperparams.transforms]
        return T.Compose(transforms)
