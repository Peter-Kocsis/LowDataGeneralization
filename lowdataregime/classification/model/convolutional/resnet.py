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
from lowdataregime.classification.model.convolutional.resnet_backbone import ResNetBackboneDefinition, \
    ResNet18BackboneHyperParameterSet, ResNet18BackboneDefinition, ResNet34BackboneDefinition, \
    ResNet34BackboneHyperParameterSet, ResNet50BackboneDefinition, ResNet50BackboneHyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace


class ResNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the ResNet"""

    def __init__(self,
                 backbone_definition: ResNetBackboneDefinition = None,
                 output_size: int = None,
                 feature_size: int = 512,
                 gradient_multiplier: float = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 head_bias: bool = False,
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param block_type: The type of the blocks
        :param num_blocks_per_layer: The number of blocks in each layer
        :param in_channels: The number of input channels
        :func:`~ClassificationModuleHyperParameterSet.__init__`
        """
        super().__init__(optimizer_definition, scheduler_definition, loss_calc_params, **kwargs)
        self.backbone_definition = backbone_definition
        self.output_size = output_size
        self.feature_size = feature_size
        self.gradient_multiplier = gradient_multiplier
        self.head_bias = head_bias


class ResNetDefinition(ClassificationModuleDefinition):
    pass


class ResNet(ClassificationModule):
    """
    ResNet architecture, which is implemented as a backbone for feature extraction
    and a fully-connected layer for classification
    """

    def __init__(self, params: ResNetHyperParameterSet = ResNetHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        backbone = self.params.backbone_definition.instantiate()

        classifier = nn.Sequential(
            OrderedDict([
                ('gradient_gate', GradientGate(gradient_multiplier=self.params.gradient_multiplier)),
                ('classifier_layers', nn.Linear(self.params.feature_size, self.params.output_size, bias=self.params.head_bias))]))

        return nn.Sequential(
            OrderedDict([
                ('backbone', backbone),
                ('classifier', classifier)]))

    @property
    def backbone(self):
        return self.model.backbone

    def forward(self, x):
        x, _ = self.features(x)
        x = self.head(x)
        return x

    def forward_with_intermediate(self, x):
        x, intermediate = self.features(x)
        x = self.head(x)
        return x, intermediate

    def features(self, x):
        x, intermediate = self.model.backbone(x)
        return x, intermediate

    @property
    def feature_model(self):
        return self.model.backbone

    def head(self, x):
        return self.model.classifier(x)

    def initialize_model(self):
        pass

    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        return parser


class ResNet18HyperParameterSet(ResNetHyperParameterSet):
    """HyperParameterSet of the ResNet18"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 intermediate_layers_to_return: List[str] = [],
                 head_bias: bool = True,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(ResNet18BackboneDefinition(
            ResNet18BackboneHyperParameterSet(
                initial_kernel_reduced=initial_kernel_reduced,
                intermediate_layers_to_return=intermediate_layers_to_return,
                optimizer_definition=optimizer_definition,
                scheduler_definition=scheduler_definition
            )),
            output_size=output_size,
            head_bias=head_bias,
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return ResNet18HyperParameterSpace(self)


class ResNet18Definition(ClassificationModuleDefinition):
    """Definition of the ResNet18"""

    def __init__(self, hyperparams: ResNet18HyperParameterSet = ResNet18HyperParameterSet()):
        super().__init__(ModelType.ResNet18, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet18

    def definition_space(self):
        return ResNet18DefinitionSpace(self.hyperparams.definition_space())


class ResNet18(ResNet):
    def __init__(self, params: ResNet18HyperParameterSet = ResNet18HyperParameterSet()):
        super().__init__(params)


class ResNet18HyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet18"""
    pass


class ResNet18DefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet18"""

    def __init__(self, hyperparam_space: ResNet18HyperParameterSpace = ResNet18HyperParameterSpace()):
        super().__init__(ModelType.ResNet18, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ResNet18Definition:
        return ResNet18Definition(self.hyperparam_space.suggest(trial))


class ResNet34HyperParameterSet(ResNetHyperParameterSet):
    """HyperParameterSet of the ResNet34"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 intermediate_layers_to_return: List[str] = [],
                 head_bias: bool = True,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(ResNet34BackboneDefinition(
            ResNet34BackboneHyperParameterSet(
                initial_kernel_reduced=initial_kernel_reduced,
                intermediate_layers_to_return=intermediate_layers_to_return,
                optimizer_definition=optimizer_definition,
                scheduler_definition=scheduler_definition
            )),
            output_size=output_size,
            head_bias=head_bias,
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return ResNet34HyperParameterSpace(self)


class ResNet34Definition(ClassificationModuleDefinition):
    """Definition of the ResNet34"""

    def __init__(self, hyperparams: ResNet34HyperParameterSet = ResNet34HyperParameterSet()):
        super().__init__(ModelType.ResNet34, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet34

    def definition_space(self):
        return ResNet34DefinitionSpace(self.hyperparams.definition_space())


class ResNet34(ResNet):
    def __init__(self, params: ResNet34HyperParameterSet = ResNet34HyperParameterSet()):
        super().__init__(params)


class ResNet34HyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet34"""
    pass


class ResNet34DefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet34"""

    def __init__(self, hyperparam_space: ResNet34HyperParameterSpace = ResNet34HyperParameterSpace()):
        super().__init__(ModelType.ResNet34, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ResNet34Definition:
        return ResNet34Definition(self.hyperparam_space.suggest(trial))


class ResNet50HyperParameterSet(ResNetHyperParameterSet):
    """HyperParameterSet of the ResNet34"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 intermediate_layers_to_return: List[str] = [],
                 head_bias: bool = True,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(ResNet50BackboneDefinition(
            ResNet50BackboneHyperParameterSet(
                initial_kernel_reduced=initial_kernel_reduced,
                intermediate_layers_to_return=intermediate_layers_to_return,
                optimizer_definition=optimizer_definition,
                scheduler_definition=scheduler_definition
            )),
            output_size=output_size,
            head_bias=head_bias,
            feature_size=2048,
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return ResNet50HyperParameterSpace(self)


class ResNet50Definition(ClassificationModuleDefinition):
    """Definition of the ResNet34"""

    def __init__(self, hyperparams: ResNet50HyperParameterSet = ResNet50HyperParameterSet()):
        super().__init__(ModelType.ResNet50, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet50

    def definition_space(self):
        return ResNet50DefinitionSpace(self.hyperparams.definition_space())


class ResNet50(ResNet):
    def __init__(self, params: ResNet50HyperParameterSet = ResNet50HyperParameterSet()):
        super().__init__(params)


class ResNet50HyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet34"""
    pass


class ResNet50DefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet34"""

    def __init__(self, hyperparam_space: ResNet50HyperParameterSpace = ResNet50HyperParameterSpace()):
        super().__init__(ModelType.ResNet50, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ResNet50Definition:
        return ResNet50Definition(self.hyperparam_space.suggest(trial))


class ResNet18redHyperParameterSet(ResNetHyperParameterSet):
    """HyperParameterSet of the ResNet18"""
    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 intermediate_layers_to_return: List[str] = [],
                 head_bias: bool = True,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(ResNet18BackboneDefinition(
            ResNet18BackboneHyperParameterSet(
                initial_kernel_reduced=initial_kernel_reduced,
                feature_size=64,
                intermediate_layers_to_return=intermediate_layers_to_return,
                optimizer_definition=optimizer_definition,
                scheduler_definition=scheduler_definition
            )),
            output_size=output_size,
            head_bias=head_bias,
            feature_size=64,
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return ResNet18redHyperParameterSpace(self)


class ResNet18redDefinition(ClassificationModuleDefinition):
    """Definition of the ResNet18"""

    def __init__(self, hyperparams: ResNet18redHyperParameterSet = ResNet18redHyperParameterSet()):
        super().__init__(ModelType.ResNet18red, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet18red

    def definition_space(self):
        return ResNet18redDefinitionSpace(self.hyperparams.definition_space())


class ResNet18red(ResNet):
    def __init__(self, params: ResNet18redHyperParameterSet = ResNet18redHyperParameterSet()):
        super().__init__(params)


class ResNet18redHyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet18"""
    pass


class ResNet18redDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet18"""

    def __init__(self, hyperparam_space: ResNet18redHyperParameterSpace = ResNet18redHyperParameterSpace()):
        super().__init__(ModelType.ResNet18red, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ResNet18redDefinition:
        return ResNet18redDefinition(self.hyperparam_space.suggest(trial))


class ResNet18BlockRedHyperParameterSet(ResNetHyperParameterSet):
    """HyperParameterSet of the ResNet18BlockRed"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 intermediate_layers_to_return: List[str] = [],
                 head_bias: bool = True,
                 num_last_chanels: int = 512,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(ResNet18BackboneDefinition(
            ResNet18BackboneHyperParameterSet(
                num_chanels=[64, 128, 256, num_last_chanels],
                feature_size=num_last_chanels,
                initial_kernel_reduced=initial_kernel_reduced,
                intermediate_layers_to_return=intermediate_layers_to_return,
                optimizer_definition=optimizer_definition,
                scheduler_definition=scheduler_definition
            )),
            output_size=output_size,
            head_bias=head_bias,
            feature_size=num_last_chanels,
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return ResNet18BlockRedHyperParameterSpace(self)


class ResNet18BlockRedDefinition(ClassificationModuleDefinition):
    """Definition of the ResNet18BlockRed"""

    def __init__(self, hyperparams: ResNet18BlockRedHyperParameterSet = ResNet18BlockRedHyperParameterSet()):
        super().__init__(ModelType.ResNet18BlockRed, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet18BlockRed

    def definition_space(self):
        return ResNet18BlockRedDefinitionSpace(self.hyperparams.definition_space())


class ResNet18BlockRed(ResNet):
    def __init__(self, params: ResNet18BlockRedHyperParameterSet = ResNet18BlockRedHyperParameterSet()):
        super().__init__(params)


class ResNet18BlockRedHyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet18BlockRed"""
    pass


class ResNet18BlockRedDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet18BlockRed"""

    def __init__(self, hyperparam_space: ResNet18BlockRedHyperParameterSpace = ResNet18BlockRedHyperParameterSpace()):
        super().__init__(ModelType.ResNet18BlockRed, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ResNet18BlockRedDefinition:
        return ResNet18BlockRedDefinition(self.hyperparam_space.suggest(trial))


class ResNet18SmallHyperParameterSet(ResNetHyperParameterSet):
    """HyperParameterSet of the ResNet18Small"""

    def __init__(self,
                 output_size: int = None,
                 gradient_multiplier: float = None,
                 intermediate_layers_to_return: List[str] = [],
                 head_bias: bool = True,
                 num_chanels: list = [64, 128, 256, 512],
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(ResNet18BackboneDefinition(
            ResNet18BackboneHyperParameterSet(
                num_chanels=num_chanels,
                feature_size=num_chanels[-1],
                initial_kernel_reduced=initial_kernel_reduced,
                intermediate_layers_to_return=intermediate_layers_to_return,
                optimizer_definition=optimizer_definition,
                scheduler_definition=scheduler_definition
            )),
            output_size=output_size,
            head_bias=head_bias,
            feature_size=num_chanels[-1],
            optimizer_definition=optimizer_definition,
            scheduler_definition=scheduler_definition,
            loss_calc_params=loss_calc_params,
            **kwargs)
        self.gradient_multiplier = gradient_multiplier

    def definition_space(self):
        return ResNet18SmallHyperParameterSpace(self)


class ResNet18SmallDefinition(ClassificationModuleDefinition):
    """Definition of the ResNet18Small"""

    def __init__(self, hyperparams: ResNet18SmallHyperParameterSet = ResNet18SmallHyperParameterSet()):
        super().__init__(ModelType.ResNet18Small, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet18Small

    def definition_space(self):
        return ResNet18SmallDefinitionSpace(self.hyperparams.definition_space())


class ResNet18Small(ResNet):
    def __init__(self, params: ResNet18SmallHyperParameterSet = ResNet18SmallHyperParameterSet()):
        super().__init__(params)


class ResNet18SmallHyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the ResNet18Small"""
    pass


class ResNet18SmallDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the ResNet18Small"""

    def __init__(self, hyperparam_space: ResNet18SmallHyperParameterSpace = ResNet18SmallHyperParameterSpace()):
        super().__init__(ModelType.ResNet18Small, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ResNet18SmallDefinition:
        return ResNet18SmallDefinition(self.hyperparam_space.suggest(trial))
