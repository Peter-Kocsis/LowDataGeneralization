from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet
from lowdataregime.utils.utils import SerializableEnum


class DistanceType(SerializableEnum):
    CORRELATION_DISTANCE = "correlation_distance"
    COSINE_DISTANCE = "cosine_distance"
    EUCLIDEAN_DISTANCE = "euclidean_distance"


class Distance(ABC):
    @abstractmethod
    def get_distance_matrix(self, x, y):
        pass


class DistanceDefinitionSet(DefinitionSet):
    """Definition of the GeneralNet"""
    pass


class CorrelationDistanceDefinitionSet(DistanceDefinitionSet):
    """Definition of the GeneralNet"""

    def __init__(self):
        super().__init__(DistanceType.CORRELATION_DISTANCE)

    def instantiate(self, *args, **kwargs):
        return CorrelationDistance()


class CorrelationDistance(Distance):
    """
    Negative shifted cosine similarity
    """
    def get_distance_matrix(self, x, y):
        x = (x - x.mean(dim=0))
        x = x / x.norm(dim=1).unsqueeze(1)

        y = (y - y.mean(dim=0))
        y = y / y.norm(dim=1).unsqueeze(1)
        return - torch.mm(x, y.t())


class CosineDistanceDefinitionSet(DistanceDefinitionSet):
    """Definition of the GeneralNet"""

    def __init__(self):
        super().__init__(DistanceType.COSINE_DISTANCE)

    def instantiate(self, *args, **kwargs):
        return CosineDistance()


class CosineDistance(Distance):
    """
    Negative Cosine similarity
    """
    def get_distance_matrix(self, x, y):
        x = x / x.norm(dim=1).unsqueeze(1)
        y = y / y.norm(dim=1).unsqueeze(1)
        return 1 - torch.mm(x, y.t())


class EuclideanDistanceDefinitionSet(DistanceDefinitionSet):
    """Definition of the GeneralNet"""

    def __init__(self):
        super().__init__(DistanceType.EUCLIDEAN_DISTANCE)

    def instantiate(self, *args, **kwargs):
        return EuclideanDistance()


class EuclideanDistance(Distance):
    def get_distance_matrix(self, x, y):
        return torch.cdist(x, y, p=2)

