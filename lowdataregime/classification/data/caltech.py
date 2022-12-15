import copy
from typing import Any, Optional, Callable, Dict, Sequence

import optuna
from torch import Generator
from torch.utils.data import Subset, random_split

from lowdataregime.classification.data._dataset._caltech import Caltech101, Caltech256
from lowdataregime.classification.data._dataset._dataset import balanced_random_ratio_split
from lowdataregime.classification.data.cifar import CIFARDataModule
from lowdataregime.classification.data.datas import DataModuleType, DataModuleDefinition
from lowdataregime.classification.sampling.sampler import SamplerDefinition
from lowdataregime.classification.data.transform import TransformDefinition, ComposeTransformDefinition, \
    ComposeTransformHyperParameterSet, RandomHorizontalFlipTransformDefinition, RandomCropTransformDefinition, \
    RandomCropTransformHyperParameterSet, ToTensorTransformDefinition, NormalizeTransformDefinition, \
    NormalizeHyperParameterSet, ResizeTransformDefinition, ResizeHyperParameterSet, RepeatTransformDefinition, \
    RepeatHyperParameterSet
from lowdataregime.classification.model.classification_module import TrainStage
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace, DefinitionSpace


class CaltechHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the CIFARDataModule"""

    def __init__(self,
                 dataset_to_use: DataModuleType = None,
                 data_dir: str = "./data",
                 val_ratio: float = 0.0,
                 num_workers: int = 0,
                 batch_size: int = 128,
                 num_classes: int = 101,
                 duplication_factor: int = 1,
                 train_samples_per_class: int = 15,
                 test_samples_per_class: float = 0.3,
                 train_transforms_def: Optional[TransformDefinition] = None,
                 val_transforms_def: Optional[TransformDefinition] = None,
                 test_transforms_def: Optional[TransformDefinition] = None,
                 train_sampler_def: Optional[SamplerDefinition] = None,
                 val_sampler_def: Optional[SamplerDefinition] = None,
                 test_sampler_def: Optional[SamplerDefinition] = None,
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param dataset_to_use: The type of the dataset to use
        :param data_dir: The root directory of the dataset
        :param val_ratio: The ratio of the the validation set to the whole training set
        :param num_workers: The number of workers used by the DataLoaders
        :param batch_size: The batch size
        :param num_classes: The number of classes in the dataset
        :param train_transforms_def: The definition of the transformations of the training set
        :param val_transforms_def: The definition of the transformations of the validation set
        :param test_transforms_def: The definition of the transformations of the test set
        :param train_sampler_def: The definition of the sampler of the training set
        :param val_sampler_def: The definition of the sampler of the validation set
        :param test_sampler_def: The definition of the sampler of the test set
        """
        super().__init__(**kwargs)
        self.duplication_factor = duplication_factor

        self.dataset_to_use = dataset_to_use
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_samples_per_class = train_samples_per_class
        self.test_samples_per_class = test_samples_per_class
        self.train_transforms_def = train_transforms_def
        self.val_transforms_def = val_transforms_def
        self.test_transforms_def = test_transforms_def
        self.train_sampler_def = train_sampler_def
        self.val_sampler_def = val_sampler_def
        self.test_sampler_def = test_sampler_def


class CaltechDataModule(CIFARDataModule):
    __Caltech_Datasets = {DataModuleType.CALTECH101: Caltech101,
                          DataModuleType.CALTECH256: Caltech256}

    def __init__(self, params: CaltechHyperParameterSet = CaltechHyperParameterSet()):
        """
        Initialize a new object
        :param dataset_to_use: Defines which CIFAR datamodule_type to use
        :param data_dir: The root of the datamodule_type folder
        :param val_ratio: The ration of the validation set to the training set
        :param num_workers: The number of workers used by the DataLoader
        :param normalize: If True adds normalize transform
        :param batch_size: The batch size
        :param args: Additional args
        :param kwargs: Additional kwargs
        """
        super().__init__()
        super(CaltechDataModule, self).__init__(params)

        self.dims = (3, 224, 224)
        self.train_samples_per_class = params.train_samples_per_class
        self.test_samples_per_class = params.test_samples_per_class

        self._dataset: Optional[Caltech101] = None
        self._dataset_rem: Optional[Caltech101] = None

    @property
    def class_to_idx(self):
        return {c: i for i, c in enumerate(self.dataset.categories)}

    @property
    def classes(self):
        return self.dataset.categories

    @property
    def original_train_len(self):
        return self.train_len

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            self._split_train_test()
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, dataset_train):
        self._dataset_train = dataset_train
        self.logger.info(f"Training dataset set to {dataset_train}, number of images: {len(dataset_train)}")

    @property
    def targets_train(self):
        if isinstance(self.dataset_train, Subset):
            return [self.dataset_train.dataset.y[idx] for idx in self.dataset_train.indices]
        return self.dataset_train.y

    @property
    def targets_test(self):
        if isinstance(self.dataset_test, Subset):
            return [self.dataset_test.dataset.y[idx] for idx in self.dataset_test.indices]
        return self.dataset_test.targets

    @property
    def dataset_valid(self):
        if self._dataset_valid is None:
            self._dataset_valid = self._load_dataset(mode=TrainStage.Validation)
        return self._dataset_valid

    @dataset_valid.setter
    def dataset_valid(self, dataset_valid):
        self._dataset_valid = dataset_valid
        self.logger.info(f"Validation dataset set to {dataset_valid}, number of images: {len(dataset_valid)}")

    @property
    def dataset_test(self):
        if self._dataset_test is None:
            self._split_train_test()
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, dataset_test):
        raise NotImplementedError()

    def _split_train_test(self):
        # if self.train_samples_per_class is not None:
            # self._dataset_test, self._dataset_train, self._dataset_rem = \
            #     balanced_random_ratio_split(self.dataset, [self.test_samples_per_class, self.train_samples_per_class], 0)
        # else:
        self._dataset_test, self._dataset_train = \
                balanced_random_ratio_split(self.dataset, self.test_samples_per_class, 0)
        self._dataset_train = copy.deepcopy(self._dataset_train)  # To ensure mutability

        self._dataset_train.dataset.transform = self.train_transforms
        self._dataset_test.dataset.transform = self.test_transforms

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_dataset(self, **kwargs):
        self.logger.debug(f"Loading dataset!")
        dataset = self.get_dataset(**kwargs)
        self.logger.debug(f"The dataset loaded!")
        return dataset

    def get_dataset(self, **kwargs):
        dataset = self.__Caltech_Datasets[self.dataset_to_use](root='./data', download=True, **kwargs)
        return dataset

    @property
    def unaugmented_dataset_train(self):
        unaugmented_dataset = copy.deepcopy(self.dataset_train)
        unaugmented_dataset.dataset.transform = self.test_transforms
        return unaugmented_dataset

    def setup(self, stage: Optional[str] = None):
        self.logger.debug(f"Setup data!")

        if self._dataset_valid is None:
            # No validation set has defined yet
            self.logger.debug(f"Splitting training data in {self.train_len} - {self.valid_len} splits!")
            if self.val_ratio is None:
                dataset_train_split = Subset(self._dataset_train, list(range(len(self._dataset_train))))
                dataset_val_split = copy.deepcopy(dataset_train_split)
            else:
                if self.valid_len == 0:
                    dataset_train_split = Subset(self._dataset_train, list(range(len(self._dataset_train))))
                    dataset_val_split = Subset(self._dataset_train, [])
                else:
                    dataset_train_split, dataset_val_split = random_split(self._dataset_train,
                                                                          [self.train_len, self.valid_len],
                                                                          generator=Generator().manual_seed(0))

                dataset_val_split.dataset = copy.deepcopy(self._dataset_train)
            self.dataset_train = dataset_train_split
            self._dataset_valid = dataset_val_split

            self.logger.debug(f"Training data split!")
        else:
            self.logger.debug(f"Validation set already defined as {self.dataset_valid}!")

        if self.filter_train is not None:
            self.logger.debug(f"Filtering training data!")
            self.dataset_train = self.filter_train.filter_dataset(self.dataset_train)

        if self.filter_valid is not None:
            self.logger.debug(f"Filtering validation data!")
            self.dataset_valid = self.filter_valid.filter_dataset(self.dataset_valid)

        self._has_setup_fit = True
        self._has_setup_test = True
        self.logger.debug(f"Data set up!")


class Caltech101HyperParameterSet(CaltechHyperParameterSet):
    def __init__(self,
                 data_dir: str = "./data",
                 val_ratio: float = 0.0,
                 num_workers: int = 0,
                 batch_size: int = 128,
                 train_samples_per_class: int = 15,
                 test_samples_per_class: float = 0.3,
                 train_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=224, padding=16)),
                         ToTensorTransformDefinition(),
                         RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=[0.5043, 0.4864, 0.4607], std=[0.3310, 0.3240, 0.3323]))
                     ])),
                 val_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=224, padding=16)),
                         ToTensorTransformDefinition(),
                         RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=[0.5091, 0.4915, 0.4645], std=[0.3337, 0.3282, 0.3373]))
                     ])),
                 test_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                         ToTensorTransformDefinition(),
                         RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=[0.5043, 0.4864, 0.4607], std=[0.3310, 0.3240, 0.3323]))
                     ])),
                 train_sampler_def: Optional[SamplerDefinition] = None,
                 val_sampler_def: Optional[SamplerDefinition] = None,
                 test_sampler_def: Optional[SamplerDefinition] = None,
                 **kwargs: Any):
        super().__init__(dataset_to_use=DataModuleType.CALTECH101,
                         data_dir=data_dir,
                         val_ratio=val_ratio,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         num_classes=101,
                         train_samples_per_class=train_samples_per_class,
                         test_samples_per_class=test_samples_per_class,
                         train_transforms_def=train_transforms_def,
                         val_transforms_def=val_transforms_def,
                         test_transforms_def=test_transforms_def,
                         train_sampler_def=train_sampler_def,
                         val_sampler_def=val_sampler_def,
                         test_sampler_def=test_sampler_def,
                         **kwargs)

    def definition_space(self):
        return Caltech101HyperParameterSpace(self)


class Caltech101Definition(DataModuleDefinition):
    def __init__(self, hyperparams: Caltech101HyperParameterSet = Caltech101HyperParameterSet()):
        super().__init__(DataModuleType.CALTECH101, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return Caltech101DataModule

    def definition_space(self):
        return Caltech101DefinitionSpace(self.hyperparams.definition_space())


class Caltech101DataModule(CaltechDataModule):
    pass


class Caltech256HyperParameterSet(CaltechHyperParameterSet):
    def __init__(self,
                 data_dir: str = "./data",
                 val_ratio: float = 0.0,
                 num_workers: int = 0,
                 batch_size: int = 128,
                 train_samples_per_class: int = 60,
                 test_samples_per_class: float = 0.3,
                 train_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=224, padding=16)),
                         ToTensorTransformDefinition(),
                         RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=[0.5091, 0.4915, 0.4645], std=[0.3337, 0.3282, 0.3373]))
                     ])),
                 val_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=224, padding=16)),
                         ToTensorTransformDefinition(),
                         RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=[0.5091, 0.4915, 0.4645], std=[0.3337, 0.3282, 0.3373]))
                     ])),
                 test_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ResizeTransformDefinition(ResizeHyperParameterSet(size=(224, 224))),
                         ToTensorTransformDefinition(),
                         RepeatTransformDefinition(RepeatHyperParameterSet(desired_num_of_channels=3)),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=[0.5091, 0.4915, 0.4645], std=[0.3337, 0.3282, 0.3373]))
                     ])),
                 train_sampler_def: Optional[SamplerDefinition] = None,
                 val_sampler_def: Optional[SamplerDefinition] = None,
                 test_sampler_def: Optional[SamplerDefinition] = None,
                 **kwargs: Any):
        super().__init__(dataset_to_use=DataModuleType.CALTECH256,
                         data_dir=data_dir,
                         val_ratio=val_ratio,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         num_classes=257,
                         train_samples_per_class=train_samples_per_class,
                         test_samples_per_class=test_samples_per_class,
                         train_transforms_def=train_transforms_def,
                         val_transforms_def=val_transforms_def,
                         test_transforms_def=test_transforms_def,
                         train_sampler_def=train_sampler_def,
                         val_sampler_def=val_sampler_def,
                         test_sampler_def=test_sampler_def,
                         **kwargs)

    def definition_space(self):
        return Caltech256HyperParameterSpace(self)

class Caltech256Definition(DataModuleDefinition):
    def __init__(self, hyperparams: Caltech256HyperParameterSet = Caltech256HyperParameterSet()):
        super().__init__(DataModuleType.CALTECH256, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return Caltech256DataModule

    def definition_space(self):
        return Caltech256DefinitionSpace(self.hyperparams.definition_space())


class Caltech256DataModule(CaltechDataModule):
    pass


class Caltech101HyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparemet_set: Caltech101HyperParameterSet = Caltech101HyperParameterSet()):
        self.default_hyperparemet_set = default_hyperparemet_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {"batch_size": [32, 64, 128, 192]}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {"batch_size": [32, 256]}

    def suggest(self, trial: optuna.Trial) -> Caltech101HyperParameterSet:
        batch_size = trial.suggest_int("batch_size", 32, 192)
        hyperparam_set = self.default_hyperparemet_set
        hyperparam_set.batch_size = batch_size
        return hyperparam_set


class Caltech101DefinitionSpace(DefinitionSpace):
    def __init__(self, hyperparam_space: Caltech101HyperParameterSpace = Caltech101HyperParameterSpace()):
        super().__init__(DataModuleType.CALTECH101, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> Caltech101Definition:
        return Caltech101Definition(self.hyperparam_space.suggest(trial))


class Caltech256HyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparemet_set: Caltech256HyperParameterSet = Caltech256HyperParameterSet()):
        self.default_hyperparemet_set = default_hyperparemet_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {"batch_size": [32, 64, 128, 192]}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {"batch_size": [32, 256]}

    def suggest(self, trial: optuna.Trial) -> Caltech256HyperParameterSet:
        batch_size = trial.suggest_int("batch_size", 32, 192)
        hyperparam_set = self.default_hyperparemet_set
        hyperparam_set.batch_size = batch_size
        return hyperparam_set


class Caltech256DefinitionSpace(DefinitionSpace):
    def __init__(self, hyperparam_space: Caltech256HyperParameterSpace = Caltech256HyperParameterSpace()):
        super().__init__(DataModuleType.CALTECH256, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> Caltech256Definition:
        return Caltech256Definition(self.hyperparam_space.suggest(trial))