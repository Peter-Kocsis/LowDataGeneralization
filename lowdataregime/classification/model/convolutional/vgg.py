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
from lowdataregime.classification.model.convolutional.vgg_backbone import VGG11BackboneDefinition, \
    VGG11BackboneHyperParameterSet, VGGBackboneDefinition
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace


class VGGHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the VGG"""

    def __init__(self,
                 feature_size: float = 512,
                 gradient_multiplier: float = None,
                 backbone_definition: VGGBackboneDefinition = None,
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
        self.feature_size = feature_size
        self.gradient_multiplier = gradient_multiplier
        self.backbone_definition = backbone_definition


class VGGDefinition(ClassificationModuleDefinition):
    pass


class VGG(ClassificationModule):
    """
    VGG architecture, which is implemented as a backbone for feature extraction
    and a fully-connected layer for classification
    Adapted from https://github.com/chengyangfu/pytorch-vgg-cifar10
    """

    def __init__(self, params: VGGHyperParameterSet = VGGHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        classifier_layers = nn.Sequential(
            nn.Linear(512, self.params.feature_size),
            nn.ReLU(True),
            nn.Linear(self.params.feature_size, self.params.feature_size),
            nn.ReLU(True),
            nn.Linear(self.params.feature_size, 10)
        )

        backbone = self.params.backbone_definition.instantiate()

        classifier = nn.Sequential(
            OrderedDict([
                ('gradient_gate', GradientGate(gradient_multiplier=self.params.gradient_multiplier)),
                ('classifier_layers', classifier_layers)]))

        return nn.Sequential(
            OrderedDict([
                ('backbone', backbone),
                ('classifier', classifier)]))

    def initialize_model(self):
        # Initialize weights
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

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


class VGG11HyperParameterSet(VGGHyperParameterSet):
    """HyperParameterSet of the VGG18"""

    def __init__(self,
                 feature_size: float = 512,
                 batch_norm: bool = False,
                 gradient_multiplier: float = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(
            feature_size=feature_size,
            gradient_multiplier=gradient_multiplier,
            backbone_definition=VGG11BackboneDefinition(
                VGG11BackboneHyperParameterSet(
                    batch_norm=batch_norm,
                    optimizer_definition=optimizer_definition,
                    scheduler_definition=scheduler_definition,
                    loss_calc_params=loss_calc_params
                )),
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return VGG11HyperParameterSpace(self)


class VGG11Definition(ClassificationModuleDefinition):
    """Definition of the ResNet18"""

    def __init__(self, hyperparams: VGG11HyperParameterSet = VGG11HyperParameterSet()):
        super().__init__(ModelType.VGG11, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return VGG11

    def definition_space(self):
        return VGG11DefinitionSpace(self.hyperparams.definition_space())


class VGG11(VGG):
    def __init__(self, params: VGG11HyperParameterSet = VGG11HyperParameterSet()):
        super().__init__(params)


class VGG11HyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet18"""
    pass


class VGG11DefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet18"""

    def __init__(self, hyperparam_space: VGG11HyperParameterSpace = VGG11HyperParameterSpace()):
        super().__init__(ModelType.VGG11, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> VGG11Definition:
        return VGG11Definition(self.hyperparam_space.suggest(trial))
