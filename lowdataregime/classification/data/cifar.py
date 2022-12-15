"""
This module contains CIFAR datamodule_type relevant implementations
"""
import copy
import functools
from argparse import ArgumentParser
from typing import Optional, Any, Callable, Sequence, Dict

import optuna
from torch import Generator
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from lowdataregime.classification.data._dataset._cifar_10_c import CIFAR10C_CorruptionType, CIFAR10C
from lowdataregime.classification.data._dataset._dataset import DuplicateDataset
from lowdataregime.classification.data._dataset.kdcl_cifar10 import KDCL_CIFAR10
from lowdataregime.classification.data._dataset.kdcl_cifar100 import KDCL_CIFAR100
from lowdataregime.classification.data.classification_datamodule import ClassificationDataModule
from lowdataregime.classification.data.datas import DataModuleDefinition, DataModuleType
from lowdataregime.classification.sampling.sampler import SamplerDefinition
from lowdataregime.classification.data.transform import TransformDefinition, ComposeTransformDefinition, \
    ComposeTransformHyperParameterSet, RandomHorizontalFlipTransformDefinition, RandomCropTransformDefinition, \
    RandomCropTransformHyperParameterSet, ToTensorTransformDefinition, NormalizeTransformDefinition, \
    NormalizeHyperParameterSet
from lowdataregime.classification.log.logger import init_logger
from lowdataregime.classification.model.classification_module import TrainStage
from lowdataregime.parameters.params import HyperParameterSpace, HyperParameterSet, DefinitionSpace
from lowdataregime.utils.utils import pl_worker_init_function


class CIFARHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the CIFARDataModule"""

    def __init__(self,
                 dataset_to_use: DataModuleType = None,
                 data_dir: str = "./data",
                 val_ratio: float = 0.2,
                 num_workers: int = 0,
                 batch_size: int = 100,
                 num_classes: int = 10,
                 duplication_factor: int = 1,
                 use_kdcl_sampling: bool = False,
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
        self.dataset_to_use = dataset_to_use
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.duplication_factor = duplication_factor
        self.use_kdcl_sampling = use_kdcl_sampling
        self.train_transforms_def = train_transforms_def
        self.val_transforms_def = val_transforms_def
        self.test_transforms_def = test_transforms_def
        self.train_sampler_def = train_sampler_def
        self.val_sampler_def = val_sampler_def
        self.test_sampler_def = test_sampler_def


class CIFARDataModule(ClassificationDataModule):
    """
    Datamodule for the CIFAR datasets
    """

    __CIFAR_Datasets = {DataModuleType.CIFAR10: CIFAR10,
                        DataModuleType.CIFAR100: CIFAR100}

    __KDCL_Datasets = {DataModuleType.CIFAR10: KDCL_CIFAR10,
                       DataModuleType.CIFAR100: KDCL_CIFAR100}

    def __init__(self, params: CIFARHyperParameterSet = CIFARHyperParameterSet()):
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
        self.logger = init_logger(self.name())

        self.dims = (3, 32, 32)
        self.data_dir = params.data_dir
        self.val_ratio = params.val_ratio
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.dataset_to_use = params.dataset_to_use
        self.num_classes = params.num_classes
        self.duplication_factor = params.duplication_factor
        self.params = params

        self.filter_train = None
        self.filter_valid = None

        self._dataset_train = None
        self._dataset_valid = None
        self._dataset_test = None
        self._dataset_robustness = None

        self._train_sampler = None
        self._val_sampler = None
        self._test_sampler = None

        self._train_transforms = None
        self._val_transforms = None
        self._test_transforms = None

        self.train_transforms = params.train_transforms_def.instantiate() if params.train_transforms_def is not None else None
        self.val_transforms = params.val_transforms_def.instantiate() if params.val_transforms_def is not None else None
        self.test_transforms = params.test_transforms_def.instantiate() if params.test_transforms_def is not None else None

        self.train_sampler = params.train_sampler_def.instantiate() if params.train_sampler_def is not None else None
        self.val_sampler = params.val_sampler_def.instantiate() if params.val_sampler_def is not None else None
        self.test_sampler = params.test_sampler_def.instantiate() if params.test_sampler_def is not None else None

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Defines the arguments of the class
        :param parent_parser: The parser which should be extended
        :return: The extended parser
        """
        super_parser = super(cls, cls).add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        parser.add_argument('--dataset_to_use', type=DataModuleType, choices=list(DataModuleType),
                            default=DataModuleType.CIFAR10)
        return parser

    @property
    def class_to_idx(self):
        return self.dataset_test.class_to_idx

    @property
    def classes(self):
        return self.dataset_test.classes

    @property
    def train_len(self):
        if self.val_ratio is None:
            return len(self.dataset_train)
        return int((1 - self.val_ratio) * len(self.dataset_train))

    @property
    def original_train_len(self):
        if isinstance(self.dataset_train, Subset):
            return self.dataset_train.dataset.original_length
        else:
            return self.dataset_train.original_length

    @property
    def valid_len(self):
        if self.val_ratio is None:
            return len(self.dataset_train)
        return int(self.val_ratio * len(self.dataset_train))

    @property
    def test_len(self):
        return len(self.dataset_test)

    @property
    def train_sampler(self):
        return self._train_sampler

    @train_sampler.setter
    def train_sampler(self, train_sampler):
        self._train_sampler = train_sampler
        self.logger.info(
            f"Train sampler set to {train_sampler}, number of images: {len(train_sampler) if train_sampler is not None else None}")

    @property
    def val_sampler(self):
        return self._val_sampler

    @val_sampler.setter
    def val_sampler(self, val_sampler):
        self._val_sampler = val_sampler
        self.logger.info(
            f"Validation sampler set to {val_sampler}, number of images: {len(val_sampler) if val_sampler is not None else None}")

    @property
    def test_sampler(self):
        return self._test_sampler

    @test_sampler.setter
    def test_sampler(self, test_sampler):
        self._test_sampler = test_sampler
        self.logger.info(
            f"Test sampler set to {test_sampler}, number of images: {len(test_sampler) if test_sampler is not None else None}")

    @property
    def train_transforms(self):
        return self._train_transforms

    @train_transforms.setter
    def train_transforms(self, train_transforms):
        self._train_transforms = train_transforms
        self.logger.info(f"Train transforms set to {train_transforms}")

    @property
    def val_transforms(self):
        return self._val_transforms

    @val_transforms.setter
    def val_transforms(self, val_transforms):
        self._val_transforms = val_transforms
        self.logger.info(f"validation transforms set to {val_transforms}")

    @property
    def test_transforms(self):
        return self._test_transforms

    @test_transforms.setter
    def test_transforms(self, test_transforms):
        self._test_transforms = test_transforms
        self.logger.info(f"Test transforms set to {test_transforms}")

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            self._dataset_train = self._load_dataset(mode=TrainStage.Training)
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, dataset_train):
        self._dataset_train = dataset_train
        self.logger.info(f"Training dataset set to {dataset_train}, number of images: {len(dataset_train)}")

    @property
    def targets_train(self):
        if isinstance(self.dataset_train, Subset):
            return [self.dataset_train.dataset.targets[idx] for idx in self.dataset_train.indices]
        return self.dataset_train.targets

    @property
    def targets_test(self):
        if isinstance(self.dataset_test, Subset):
            return [self.dataset_test.dataset.targets[idx] for idx in self.dataset_test.indices]
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
            self._dataset_test = self._load_dataset(mode=TrainStage.Test)
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, dataset_test):
        self._dataset_test = dataset_test
        self.logger.info(f"Test dataset set to {dataset_test}, number of images: {len(dataset_test)}")

    @property
    def dataset_robustness(self):
        if self._dataset_robustness is None:
            self._dataset_robustness = self._load_dataset(mode=TrainStage.Robustness, robustness=True)
        return self._dataset_robustness

    @dataset_robustness.setter
    def dataset_robustness(self, dataset_robustness):
        self._dataset_robustness = dataset_robustness
        self.logger.info(f"Robustness dataset set to {dataset_robustness}, number of images: {len(dataset_robustness)}")

    def get_transform(self, mode: TrainStage):
        if mode == TrainStage.Training:
            return self.train_transforms
        if mode == TrainStage.Validation:
            return self.val_transforms
        if mode == TrainStage.Test:
            return self.test_transforms
        if mode == TrainStage.Robustness:
            return self.test_transforms
        raise RuntimeError(f"Unknown stage: {mode}")

    def _load_dataset(self, mode: TrainStage, **kwargs):
        train = mode.is_train()
        self.logger.debug(f"Loading {mode} dataset! - training: {train}")
        kwargs['transform'] = self.get_transform(mode)
        dataset = self.get_dataset(train, **kwargs)
        self.logger.debug(f"The {mode} dataset loaded!")
        return dataset

    @property
    def unaugmented_dataset_train(self):
        return self.get_dataset(train=True, transform=self.get_transform(TrainStage.Test))

    def get_dataset(self, train: bool, robustness: bool = False, **kwargs):
        if self.params.use_kdcl_sampling:
            dataset_map = self.__KDCL_Datasets
        else:
            dataset_map = self.__CIFAR_Datasets

        if robustness:
            assert self.dataset_to_use == DataModuleType.CIFAR10, "Only CIFAR10 Robustness training is defined!"
            dataset = {corruption_type: {corruption_severity: CIFAR10C(root='./data',
                                 train=train,
                                 download=True,
                                 corruption_type=corruption_type,
                                 corruption_severity=corruption_severity,
                                 **kwargs)
                        for corruption_severity in range(5)} for corruption_type in CIFAR10C_CorruptionType}
        else:
            dataset = dataset_map[self.dataset_to_use](root='./data', train=train, download=True, **kwargs)
        if train:
            dataset = DuplicateDataset(dataset=dataset, duplication_factor=self.duplication_factor)
        return dataset

    def prepare_data(self):
        self.logger.debug(f"Preparing data!")
        if self.dataset_train is None or self.dataset_test is None:
            raise RuntimeError("Unable to load the datamodule_type!")
        self.logger.debug(f"Data prepared!")

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
            self.dataset_valid = dataset_val_split

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

    def train_dataloader(self):
        self.logger.debug(f"Creating train dataloader!")
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=self.train_sampler,
            worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
        )
        self.logger.debug(
            f"Train dataloader created with dataset length: {len(loader.dataset)}, sampler length: {len(loader.sampler) if loader.sampler is not None else None}!")
        return loader

    def val_dataloader(self):
        self.logger.debug(f"Creating validation dataloader!")
        if len(self.dataset_valid) > 0:
            loader = DataLoader(
                self.dataset_valid,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=self.val_sampler,
                worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
            )
        else:
            self.logger.debug(f"Validation dataloader is empty, using the training set!")
            loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=self.train_sampler,
                worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
            )
        self.logger.debug(
            f"Validation dataloader created: {len(loader.dataset)}, sampler length: {len(loader.sampler) if loader.sampler is not None else None}!")
        return loader

    def test_dataloader(self):
        self.logger.debug(f"Creating test dataloader!")
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=self.test_sampler,
            worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
        )
        self.logger.debug(
            f"Test dataloader created: {len(loader.dataset)}, sampler length: {len(loader.sampler) if loader.sampler is not None else None}!")
        return loader

    def robustness_dataloader(self):
        self.logger.debug(f"Creating test dataloader!")
        loader = {corruption_type:
            {courruption_severity: DataLoader(
            severity_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=self.test_sampler,
            worker_init_fn=functools.partial(pl_worker_init_function, rank=0)
        ) for courruption_severity, severity_datasets in courrupted_datasets.items()}
            for corruption_type, courrupted_datasets in self.dataset_robustness.items()}
        self.logger.debug(
            f"Robustness dataloader created: {loader}!")
        return loader


# ----------------------------------- CIFAR10 -----------------------------------


class CIFAR10HyperParameterSet(CIFARHyperParameterSet):
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    def __init__(self,
                 data_dir: str = "./data",
                 val_ratio: float = 0.2,
                 num_workers: int = 0,
                 batch_size: int = 128,
                 use_kdcl_sampling: bool = False,
                 duplication_factor: int = 1,
                 train_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=32, padding=4)),
                         ToTensorTransformDefinition(),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=MEAN, std=STD))
                     ])),
                 val_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=32, padding=4)),
                         ToTensorTransformDefinition(),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=MEAN, std=STD))
                     ])),
                 test_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ToTensorTransformDefinition(),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=MEAN, std=STD))
                     ])),
                 train_sampler_def: Optional[SamplerDefinition] = None,
                 val_sampler_def: Optional[SamplerDefinition] = None,
                 test_sampler_def: Optional[SamplerDefinition] = None,
                 **kwargs: Any):
        super().__init__(dataset_to_use=DataModuleType.CIFAR10,
                         data_dir=data_dir,
                         val_ratio=val_ratio,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         use_kdcl_sampling=use_kdcl_sampling,
                         num_classes=10,
                         duplication_factor=duplication_factor,
                         train_transforms_def=train_transforms_def,
                         val_transforms_def=val_transforms_def,
                         test_transforms_def=test_transforms_def,
                         train_sampler_def=train_sampler_def,
                         val_sampler_def=val_sampler_def,
                         test_sampler_def=test_sampler_def,
                         **kwargs)

    def definition_space(self):
        return CIFAR10HyperParameterSpace(self)


class CIFAR10Definition(DataModuleDefinition):

    def __init__(self, hyperparams: CIFAR10HyperParameterSet = CIFAR10HyperParameterSet()):
        super().__init__(DataModuleType.CIFAR10, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return CIFAR10DataModule

    def definition_space(self):
        return CIFAR10DefinitionSpace(self.hyperparams.definition_space())


class CIFAR10DataModule(CIFARDataModule):
    pass


# ---------------------------------- CIFAR100 -----------------------------------


class CIFAR100HyperParameterSet(CIFARHyperParameterSet):
    MEAN = [0.5071, 0.4867, 0.4408]
    STD = [0.2675, 0.2565, 0.2761]
    def __init__(self,
                 data_dir: str = "./data",
                 val_ratio: float = 0.2,
                 num_workers: int = 0,
                 batch_size: int = 128,
                 duplication_factor: int = 1,
                 train_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=32, padding=4)),
                         ToTensorTransformDefinition(),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=MEAN, std=STD))
                     ])),
                 val_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         RandomHorizontalFlipTransformDefinition(),
                         RandomCropTransformDefinition(RandomCropTransformHyperParameterSet(size=32, padding=4)),
                         ToTensorTransformDefinition(),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=MEAN, std=STD))
                     ])),
                 test_transforms_def: Optional[TransformDefinition] = ComposeTransformDefinition(
                     ComposeTransformHyperParameterSet([
                         ToTensorTransformDefinition(),
                         NormalizeTransformDefinition(
                             NormalizeHyperParameterSet(mean=MEAN, std=STD))
                     ])),
                 train_sampler_def: Optional[SamplerDefinition] = None,
                 val_sampler_def: Optional[SamplerDefinition] = None,
                 test_sampler_def: Optional[SamplerDefinition] = None,
                 **kwargs: Any):
        super().__init__(dataset_to_use=DataModuleType.CIFAR100,
                         data_dir=data_dir,
                         val_ratio=val_ratio,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         num_classes=100,
                         duplication_factor=duplication_factor,
                         train_transforms_def=train_transforms_def,
                         val_transforms_def=val_transforms_def,
                         test_transforms_def=test_transforms_def,
                         train_sampler_def=train_sampler_def,
                         val_sampler_def=val_sampler_def,
                         test_sampler_def=test_sampler_def,
                         **kwargs)

    def definition_space(self):
        return CIFAR100HyperParameterSpace(self)


class CIFAR100Definition(DataModuleDefinition):
    def __init__(self, hyperparams: CIFAR100HyperParameterSet = CIFAR100HyperParameterSet()):
        super().__init__(DataModuleType.CIFAR100, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return CIFAR100DataModule

    def definition_space(self):
        return CIFAR100DefinitionSpace(self.hyperparams.definition_space())


class CIFAR100DataModule(CIFARDataModule):
    pass


class CIFAR10HyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparemet_set: CIFAR10HyperParameterSet = CIFAR10HyperParameterSet()):
        self.default_hyperparemet_set = default_hyperparemet_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {"batch_size": [25, 50, 100, 200]}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {"batch_size": [25, 200]}

    def suggest(self, trial: optuna.Trial) -> CIFAR100HyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparemet_set)

        if hyperparam_set.batch_size is None:
            hyperparam_set.batch_size = trial.suggest_categorical("batch_size", [25, 50, 100, 200])
        return hyperparam_set


class CIFAR10DefinitionSpace(DefinitionSpace):
    def __init__(self, hyperparam_space: CIFAR10HyperParameterSpace = CIFAR10HyperParameterSpace()):
        super().__init__(DataModuleType.CIFAR10, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> CIFAR10Definition:
        return CIFAR10Definition(self.hyperparam_space.suggest(trial))


class CIFAR100HyperParameterSpace(CIFAR10HyperParameterSpace):
    def __init__(self, default_hyperparemet_set: CIFAR100HyperParameterSet = CIFAR100HyperParameterSet()):
        super().__init__(default_hyperparemet_set)


class CIFAR100DefinitionSpace(DefinitionSpace):
    def __init__(self, hyperparam_space: CIFAR100HyperParameterSpace = CIFAR100HyperParameterSpace()):
        super().__init__(DataModuleType.CIFAR100, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> CIFAR100Definition:
        return CIFAR10Definition(self.hyperparam_space.suggest(trial))
