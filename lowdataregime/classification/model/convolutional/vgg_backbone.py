from argparse import ArgumentParser
from collections import OrderedDict
from typing import List, Any, Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model._network.resnet_block import BasicBlock, BottleneckBlock, BlockType
from lowdataregime.classification.model.classification_module import ClassificationModule, \
    ClassificationModuleHyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition


class VGGBackboneHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the PoorVGG"""

    def __init__(self,
                 config: list = [],
                 batch_norm: bool = False,
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
        self.config = config
        self.batch_norm = batch_norm


class VGGBackboneDefinition(ClassificationModuleDefinition):
    pass


class VGGBackbone(ClassificationModule):
    """
    VGG-based architecture only for feature extraction i.e., without last fully-connected layer
    Adapted from https://github.com/chengyangfu/pytorch-vgg-cifar10
    """

    def __init__(self, params: VGGBackboneHyperParameterSet = VGGBackboneHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        layers = []
        in_channels = 3
        for idx, layer_conf in enumerate(self.params.config):
            if layer_conf == 'M':
                layers += [(f"{idx}_maxpool", nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                conv2d = nn.Conv2d(in_channels, layer_conf, kernel_size=3, padding=1)
                if self.params.batch_norm:
                    layers += [(f"{idx}_conv", conv2d),
                               (f"{idx}_bn", nn.BatchNorm2d(layer_conf)),
                               (f"{idx}_relu", nn.ReLU(inplace=True))]
                else:
                    layers += [(f"{idx}_conv", conv2d),
                               (f"{idx}_relu", nn.ReLU(inplace=True))]
                in_channels = layer_conf
        layers += [(f"{len(self.params.config)}_flatten", nn.Flatten())]
        return nn.Sequential(OrderedDict(layers))

    def initialize_model(self):
        pass

    @classmethod
    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        return parser


class VGG11BackboneHyperParameterSet(VGGBackboneHyperParameterSet):
    def __init__(self,
                 batch_norm: bool = False,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(config=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                         batch_norm=batch_norm,
                         optimizer_definition=optimizer_definition,
                         scheduler_definition=scheduler_definition,
                         loss_calc_params=loss_calc_params,
                         **kwargs)


class VGG11BackboneDefinition(VGGBackboneDefinition):
    def __init__(self, hyperparams: VGG11BackboneHyperParameterSet = VGG11BackboneHyperParameterSet()):
        super().__init__(ModelType.VGG11Backbone, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return VGG11Backbone


class VGG11Backbone(VGGBackbone):
    def __init__(self, params: VGG11BackboneHyperParameterSet = VGG11BackboneHyperParameterSet()):
        super().__init__(params)