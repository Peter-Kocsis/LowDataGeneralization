import math
from abc import ABC
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Callable, Dict, Sequence, List

import optuna
import torch
import torch.nn as nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model._network.gradient_gate import GradientGate
from lowdataregime.classification.model.classification_module import ClassificationModule, \
    ClassificationModuleHyperParameterSet, ClassificationModuleHyperParameterSpace
from lowdataregime.classification.model.convolutional.efficientnet_backbone import EfficientNetBackbone
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace


class EfficientNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the VGG"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 backbone_name: str = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param block_type: The type of the blocks
        :param num_blocks_per_layer: The number of blocks in each layer
        :param in_channels: The number of input channels
        :func:`~ClassificationModuleHyperParameterSet.__init__`
        """
        super().__init__(optimizer_definition, scheduler_definition, loss_calc_params, **kwargs)
        self.output_size = output_size
        self.gradient_multiplier = gradient_multiplier
        self.backbone_name = backbone_name


class EfficientNetDefinition(ClassificationModuleDefinition):
    pass


class EfficientNet(ClassificationModule):
    """
    EfficientNet architecture, which is implemented as a backbone for feature extraction
    and a fully-connected layer for classification
    Adapted from https://github.com/lukemelas/EfficientNet-PyTorch
    """

    def __init__(self, params: EfficientNetHyperParameterSet = EfficientNetHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        backbone = EfficientNetBackbone.from_name(self.params.backbone_name)

        classifier = nn.Sequential(
            OrderedDict([
                ('gradient_gate', GradientGate(gradient_multiplier=self.params.gradient_multiplier)),
                ('classifier_layers', nn.Linear(backbone.out_channels, self.params.output_size))]))

        return nn.Sequential(
            OrderedDict([
                ('backbone', backbone),
                ('classifier', classifier)]))

    def initialize_model(self):
        pass

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, x):
        x, _ = self.features(x)
        x = self.head(x)
        return x

    def features(self, x):
        x = self.model.backbone(x)
        return x, None

    @property
    def feature_model(self):
        return self.model.backbone

    def head(self, x):
        return self.model.classifier(x)

    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        return parser


class EfficientNetB3HyperParameterSet(EfficientNetHyperParameterSet):
    """HyperParameterSet of the EfficientNet18"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(
            output_size=output_size,
            gradient_multiplier=gradient_multiplier,
            backbone_name='efficientnet-b3_small',
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return EfficientNetB3HyperParameterSpace(self)


class EfficientNetB3Definition(ClassificationModuleDefinition):
    """Definition of the ResNet18"""

    def __init__(self, hyperparams: EfficientNetB3HyperParameterSet = EfficientNetB3HyperParameterSet()):
        super().__init__(ModelType.EfficientNetB3, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return EfficientNetB3

    def definition_space(self):
        return EfficientNetB3DefinitionSpace(self.hyperparams.definition_space())


class EfficientNetB3(EfficientNet):
    def __init__(self, params: EfficientNetB3HyperParameterSet = EfficientNetB3HyperParameterSet()):
        super().__init__(params)


class EfficientNetB3HyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet18"""
    pass


class EfficientNetB3DefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet18"""

    def __init__(self, hyperparam_space: EfficientNetB3HyperParameterSpace = EfficientNetB3HyperParameterSpace()):
        super().__init__(ModelType.EfficientNetB3, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> EfficientNetB3Definition:
        return EfficientNetB3Definition(self.hyperparam_space.suggest(trial))
