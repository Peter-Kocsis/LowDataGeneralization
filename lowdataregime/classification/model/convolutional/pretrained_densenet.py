import copy
import os
from collections import OrderedDict
from enum import Enum
from typing import Sequence, Dict

import optuna
from pretrainedmodels import resnet18, resnet34
from torchvision.datasets.utils import download_url
from torchvision.models import densenet121

from lowdataregime.classification.model.convolutional.resnet import ResNet18
from lowdataregime.classification.model.models import PretrainingType
from lowdataregime.utils.utils import SerializableEnum
from lowdataregime.classification.model._network.gradient_gate import GradientGate
from lowdataregime.classification.model.convolutional.resnet_backbone import *
from lowdataregime.parameters.params import HyperParameterSpace, \
    DefinitionSpace, HyperParameterSet


class PretrainedDenseNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the PretrainedResNet18"""

    def __init__(self,
                 pretraining_type: PretrainingType = PretrainingType.ImageNet,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None, **kwargs: Any):
        super().__init__(optimizer_definition, scheduler_definition, **kwargs)
        self.pretraining_type = pretraining_type
        self.output_size = output_size
        self.gradient_multiplier = gradient_multiplier
        self.feature_size = 1024


class PretrainedDenseNetDefinition(ClassificationModuleDefinition):
    """Definition of the PretrainedResNet18"""

    def __init__(self,
                 hyperparams: PretrainedDenseNetHyperParameterSet = PretrainedDenseNetHyperParameterSet()):
        super().__init__(ModelType.PretrainedDenseNet, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return PretrainedDenseNet


class PretrainedDenseNet(ClassificationModule):
    """ResNet-based architecture, which is implemented as a backbone for feature extraction and a fully-connected layer for classification"""
    model_path = "models"

    def __init__(self, params: PretrainedDenseNetHyperParameterSet = PretrainedDenseNetHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        if self.params.pretraining_type == PretrainingType.ImageNet:
            pretrained_model = densenet121(pretrained=True)
            pretrained_model.classifier = None
            self.layers_root = "model.backbone.pretrained_model"
        else:
            raise NotImplementedError(f"Pretraining type {self.params.pretraining_type} not implemented!")

        backbone = nn.Sequential(
            OrderedDict([
                ('pretrained_model', pretrained_model),
                ('avg_pool', nn.AdaptiveAvgPool2d(output_size=(1, 1))),
                ('flatten', nn.Flatten())
            ])
        )

        classifier = nn.Sequential(
            OrderedDict([
                ('gradient_gate', GradientGate(gradient_multiplier=self.params.gradient_multiplier)),
                ('classifier_layers', nn.Linear(self.params.feature_size, self.params.output_size))]))

        return nn.ModuleDict(
            OrderedDict([
                ('backbone', backbone),
                ('classifier', classifier)]))

    @property
    def backbone(self):
        return self.features

    def forward(self, x):
        x, _ = self.features(x)
        x = self.head(x)
        return x

    def forward_with_intermediate(self, x):
        x, intermediate = self.features(x)
        x = self.head(x)
        return x, intermediate

    @property
    def feature_model(self):
        return self.features

    def features(self, x):
        if self.params.pretraining_type == PretrainingType.ImageNet:
            x = self.model.backbone.pretrained_model.features(x)
            x = F.relu(x, inplace=True)
            x = self.model.backbone.avg_pool(x)
            x = self.model.backbone.flatten(x)
            return x, dict()
        else:
            raise NotImplementedError(f"Pretraining type {self.params.pretraining_type} not implemented!")

    def head(self, x):
        return self.model.classifier(x)

    def initialize_model(self):
        pass

    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        return parser
