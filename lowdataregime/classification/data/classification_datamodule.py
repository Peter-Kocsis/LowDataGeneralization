"""
This module contains base class of datamodule for classification tasks
"""
from abc import ABC, abstractmethod, ABCMeta
from argparse import ArgumentParser
from typing import Dict, List

from pytorch_lightning import LightningDataModule


class _ClassificationDataModuleMeta(LightningDataModule.__class__, ABCMeta):
    """Meta base class of the abstract datamodule"""
    pass


class ClassificationDataModule(LightningDataModule, ABC, metaclass=_ClassificationDataModuleMeta):
    """
    Abstract datamodule base class for classification tasks
    """

    @property
    @abstractmethod
    def class_to_idx(self) -> Dict[str, int]:
        """
        Returns a mapping from class label to class index
        :return: Dictionary of class names and index pairs
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """
        Returns the class labels
        :return: List of class labels
        """
        raise NotImplementedError()

    @classmethod
    def name(cls) -> str:
        return cls.__name__
