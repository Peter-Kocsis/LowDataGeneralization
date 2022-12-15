import copy
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import Type, Union, Dict, Sequence, Any

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import argparse_utils

from lowdataregime.classification.log.logger import init_logger
from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet, LossCalculator
from lowdataregime.classification.metrics.accuracy import TopKAccuracy
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace
from lowdataregime.utils.utils import rgetattr


class TrainStage(Enum):
    """Definition of the different training stages"""
    Training = "train"
    Validation = "valid"
    Test = "test"
    Robustness = "robustness"

    def is_train(self):
        """
        Checks whether the stage referes to a training stage or not
        :return: True if the stage is Training or Validation
        """
        return self == self.Training or self == self.Validation

    def __str__(self):
        return self.value


class ClassificationModuleHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the NormalizeTransform"""

    def __init__(self,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param optimizer_definition: The definition of the optimizer
        :param scheduler_definition: The definition of the scheduler
        """
        super().__init__(**kwargs)
        self.optimizer_definition = optimizer_definition
        self.scheduler_definition = scheduler_definition
        self.loss_calc_params = loss_calc_params


class ClassificationModule(pl.LightningModule, ABC):
    """
    Base module for classifiers
    This class should be used for all classifiers.
    It already implements the skeleton of a classification task with many useful features
    """
    def __init__(self, params: ClassificationModuleHyperParameterSet = ClassificationModuleHyperParameterSet()):
        """
        Initialize new object
        :param learning_rate: The learning rate
        """
        self.save_hyperparameters()
        super().__init__()
        self.module_logger = init_logger(self.__class__.__name__)
        self.model = self.define_model()
        self.initialize_model()
        self.train_acc = None
        self.valid_acc = None
        self.test_acc = None
        self.test_5_acc = None
        self.loss_calculator = LossCalculator(self.params.loss_calc_params)
        self.inspected_variables = {}

    @property
    def params(self):
        return self.hparams["params"]

    @abstractmethod
    def define_model(self) -> torch.nn.Module:
        """
        Define the model of the module
        :return: The model of the classifier
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_model(self):
        """
        Initialize the weights of the model
        """
        pass

    def inspect_layer_output(self, layer_path, name: str = None, storage_dict: dict = None, unsqueeze: bool = True):
        if storage_dict is None:
            storage_dict = self.inspected_variables

        if name is None:
            name = layer_path

        def hook(model, input, output):
            if unsqueeze:
                output = output.unsqueeze(0)
            if name in storage_dict:
                storage_dict[name] = torch.cat((storage_dict[name], output), dim=0)
            else:
                storage_dict[name] = output

        return rgetattr(self, layer_path).register_forward_hook(hook)

    def clean_inspected_layers(self):
        self.inspected_variables = {}

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Defines the arguments of the class
        :param parent_parser: The parser which should be extended
        :return: The extended parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @classmethod
    def from_argparse_args(cls: Type['_T'], args: Union[Namespace, ArgumentParser], **kwargs) -> '_T':
        """
        Creates new object from arguments
        """
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    def forward(self, input, *args, **kwargs):
        """
        Implements the forward pass of the model
        :param input: Input to be forwarded
        :return: The output of the model
        """
        output = self.model.forward(input)
        return output

    @classmethod
    def name(cls):
        """Returns the class name"""
        return cls.__name__

    def general_step(self, batch, batch_idx, mode: TrainStage):
        """
        General step used in all phases
        :param batch: The current batch of data
        :param batch_idx: The current batch index
        :param mode: The current phase
        :return: The loss
        """
        x, y = batch

        # register loss inspection
        self.loss_calculator.inspect_layers(self)

        # forward pass
        logits = self(x)

        # loss
        loss, sub_losses = self.loss_calculator(logits, y)

        # deregister loss inspection
        self.loss_calculator.remove_inspection()

        # accuracy
        accuracy_metric = self.get_accuracy_metric(mode)
        if isinstance(accuracy_metric, tuple):
            for metric in accuracy_metric:
                metric(logits, y)
        else:
            accuracy_metric(logits, y)

        if mode != TrainStage.Test:
            self.log(f"{mode}_loss", loss, logger=True)
            self.log(f"{mode}_acc", accuracy_metric, logger=True)
            # self.log(f"{mode}_logits", logits, logger=False, on_epoch=True, reduce_fx=lambda x: x)
            # self.log(f"{mode}_targets", y, logger=False, on_epoch=True, reduce_fx=lambda x: x)

        for key, sub_loss in sub_losses.items():
            self.log(f"{mode}_loss_{key}", sub_loss, logger=True)

        return loss

    def get_accuracy_metric(self, mode: TrainStage):
        """
        Returns the accuracy metric of the model
        :param mode: The TrainStage, whose accuracy metric is requested
        :return: The requested accuracy metric
        """
        if mode == TrainStage.Training:
            return self.train_acc
        if mode == TrainStage.Validation:
            return self.valid_acc
        if mode == TrainStage.Test:
            return self.test_acc, self.test_5_acc
        raise RuntimeError(f"Unknown stage: {mode}")

    def training_step(self, batch, batch_idx, *args):
        """Abstract definition of the training step"""
        return self.general_step(batch, batch_idx, TrainStage.Training)

    def validation_step(self, batch, batch_idx):
        """Abstract definition of the validation step"""
        return self.general_step(batch, batch_idx, TrainStage.Validation)

    def test_step(self, batch, batch_idx):
        """Abstract definition of the test step"""
        return self.general_step(batch, batch_idx, TrainStage.Test)

    def init_metrics(self):
        """Initializes the metric"""
        self.train_acc = pl.metrics.Accuracy().to(self.device)
        self.valid_acc = pl.metrics.Accuracy(compute_on_step=False).to(self.device)
        self.test_acc = pl.metrics.Accuracy(compute_on_step=False).to(self.device)
        self.test_5_acc = TopKAccuracy(top_k=5, compute_on_step=False).to(self.device)
        self.module_logger.info("Metrics initialized!")

    def configure_optimizers(self):
        """Configures the optimizers and schedulers"""
        self.init_metrics()

        self.module_logger.info(f"All params: {len(list(self.parameters()))}, Trainable: {len(list(filter(lambda p: p.requires_grad, self.parameters())))}")

        optimizer = self.params.optimizer_definition.instantiate(params=filter(lambda p: p.requires_grad, self.parameters()))
        if self.params.scheduler_definition is not None:
            scheduler = self.params.scheduler_definition.instantiate(optimizer=optimizer)
            self.module_logger.info(f"Optimizer - {self.params.optimizer_definition.type} and scheduler - {self.params.scheduler_definition.type} created!")
            return [optimizer], [scheduler]
        self.module_logger.info(f"Optimizer - {self.params.optimizer_definition.type} created!")
        return optimizer


class ClassificationModuleHyperParameterSpace(HyperParameterSpace, ABC):
    """HyperParameterSpace of the ClassificationModule"""

    def __init__(self,
                 default_hyperparam_set: ClassificationModuleHyperParameterSet = ClassificationModuleHyperParameterSet(),
                 ):
        self.default_hyperparam_set = default_hyperparam_set
        self.optimizer_space = default_hyperparam_set.optimizer_definition.definition_space() \
            if default_hyperparam_set.optimizer_definition is not None else None
        self.scheduler_space = default_hyperparam_set.scheduler_definition.definition_space() \
            if default_hyperparam_set.scheduler_definition is not None else None

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        search_grid = {}
        if self.optimizer_space is not None:
            search_grid.update(self.optimizer_space.search_grid)

        if self.scheduler_space is not None:
            search_grid.update(self.scheduler_space.search_grid)
        return search_grid

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        search_space = {}
        if self.optimizer_space is not None:
            search_space.update(self.optimizer_space.search_grid)

        if self.scheduler_space is not None:
            search_space.update(self.scheduler_space.search_grid)
        return search_space

    def suggest(self, trial: optuna.Trial) -> HyperParameterSet:
        """
        Sugges new HyperParameterSet for a trial
        :return: Suggested HyperParameterSet
        """
        hyperparams = copy.deepcopy(self.default_hyperparam_set)
        hyperparams.optimizer_definition = self._suggest_optimizer_definition(trial)
        hyperparams.scheduler_definition = self._suggest_scheduler_definition(trial)
        hyperparams.loss_calc_params = self.default_hyperparam_set.loss_calc_params

        return hyperparams

    def _suggest_optimizer_definition(self, trial: optuna.Trial):
        if self.optimizer_space is None:
            return None
        return self.optimizer_space.suggest(trial)

    def _suggest_scheduler_definition(self, trial: optuna.Trial):
        if self.scheduler_space is None:
            return None
        return self.scheduler_space.suggest(trial)
