import copy
import os
from collections import OrderedDict
from enum import Enum
from typing import Sequence, Dict

import optuna
from pretrainedmodels import resnet18, resnet34, resnet50
from torchvision.datasets.utils import download_url

from lowdataregime.classification.model.convolutional.resnet import ResNet18
from lowdataregime.classification.model.models import PretrainingType
from lowdataregime.utils.utils import SerializableEnum
from lowdataregime.classification.model._network.gradient_gate import GradientGate
from lowdataregime.classification.model.convolutional.resnet_backbone import *
from lowdataregime.parameters.params import HyperParameterSpace, \
    DefinitionSpace, HyperParameterSet


class ResNetType(SerializableEnum):
    ResNet18 = "ResNet18"
    ResNet34 = "ResNet34"
    ResNet50 = "ResNet50"


class PretrainedResNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the PretrainedResNet18"""

    def __init__(self,
                 type: ResNetType = ResNetType.ResNet18,
                 pretraining_type: PretrainingType = PretrainingType.ImageNet,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None, **kwargs: Any):
        super().__init__(optimizer_definition, scheduler_definition, **kwargs)
        self.type = type
        self.pretraining_type = pretraining_type
        self.output_size = output_size
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return PretrainedResNet18HyperParameterSpace(self)


class PretrainedResNetDefinition(ClassificationModuleDefinition):
    """Definition of the PretrainedResNet18"""

    def __init__(self,
                 hyperparams: PretrainedResNetHyperParameterSet = PretrainedResNetHyperParameterSet()):
        super().__init__(ModelType.PretrainedResNet18, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return PretrainedResNet

    def definition_space(self):
        return PretrainedResNet18DefinitionSpace(self.hyperparams.definition_space())


class PretrainedResNet(ClassificationModule):
    """ResNet-based architecture, which is implemented as a backbone for feature extraction and a fully-connected layer for classification"""
    model_path = "models"

    def __init__(self, params: PretrainedResNetHyperParameterSet = PretrainedResNetHyperParameterSet()):
        self.layers_root = None

        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        model_functions = {
            ResNetType.ResNet18: resnet18,
            ResNetType.ResNet34: resnet34,
            ResNetType.ResNet50: resnet50
        }
        model_function = model_functions[self.params.type]

        if self.params.pretraining_type == PretrainingType.ImageNet:
            pretrained_model = model_function(pretrained='imagenet')
            self.layers_root = "model.backbone.pretrained_model"
        else:
            pretrained_model = model_function(pretrained=None)

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
                ('classifier_layers', nn.Linear(512, self.params.output_size))]))

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
        intermediate = {}
        hooks = []
        hooks.append(self.inspect_layer_output(f"model.backbone.pretrained_model.layer1", name="layer1",
                                               storage_dict=intermediate, unsqueeze=False))
        hooks.append(self.inspect_layer_output(f"model.backbone.pretrained_model.layer2", name="layer2",
                                               storage_dict=intermediate, unsqueeze=False))
        hooks.append(self.inspect_layer_output(f"model.backbone.pretrained_model.layer3", name="layer3",
                                               storage_dict=intermediate, unsqueeze=False))
        hooks.append(self.inspect_layer_output(f"model.backbone.pretrained_model.layer4", name="layer4",
                                               storage_dict=intermediate, unsqueeze=False))
        x = self.model.backbone.pretrained_model.features(x)
        x = self.model.backbone.avg_pool(x)
        x = self.model.backbone.flatten(x)

        for hook in hooks:
            hook.remove()
        return x, intermediate

    def head(self, x):
        return self.model.classifier(x)

    def initialize_model(self):
        pass

    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        return parser


class PretrainedResNet18HyperParameterSpace(HyperParameterSpace):
    """HyperParameterSpace of the DummyModule"""

    def __init__(self,
                 default_hyperparam_set: PretrainedResNetHyperParameterSet = PretrainedResNetHyperParameterSet()):
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
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> HyperParameterSet:
        hyperparam_set = self.default_hyperparam_set

        hyperparam_set.optimizer_definition = self._suggest_optimizer_definition(trial)
        hyperparam_set.scheduler_definition = self._suggest_scheduler_definition(trial)

        return hyperparam_set

    def _suggest_optimizer_definition(self, trial: optuna.Trial):
        if self.optimizer_space is None:
            return None
        return self.optimizer_space.suggest(trial)

    def _suggest_scheduler_definition(self, trial: optuna.Trial):
        if self.scheduler_space is None:
            return None
        return self.scheduler_space.suggest(trial)


class PretrainedResNet18DefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self,
                 hyperparam_space: PretrainedResNet18HyperParameterSpace = PretrainedResNet18HyperParameterSpace()):
        super().__init__(ModelType.PretrainedResNet18, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> PretrainedResNetDefinition:
        return PretrainedResNetDefinition(self.hyperparam_space.suggest(trial))
