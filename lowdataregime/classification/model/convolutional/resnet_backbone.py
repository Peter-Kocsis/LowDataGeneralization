from argparse import ArgumentParser
from typing import List, Any, Callable, Dict

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


class ResNetBackboneHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the PoorResNet"""

    def __init__(self,
                 block_type: BlockType = None,
                 num_blocks_per_layer: List[int] = None,
                 in_channels: int = 3,
                 initial_kernel_reduced: bool = True,
                 intermediate_layers_to_return: List[str] = [],
                 feature_size: int = 512,
                 num_chanels: list = None,
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
        self.block_type = block_type
        self.num_blocks_per_layer = num_blocks_per_layer
        self.in_channels = in_channels
        self.initial_kernel_reduced = initial_kernel_reduced
        self.intermediate_layers_to_return = intermediate_layers_to_return
        self.feature_size = feature_size
        self.num_chanels = num_chanels


class ResNetBackboneDefinition(ClassificationModuleDefinition):
    pass


class ResNetBackbone(ClassificationModule):
    """ResNet-based architecture only for feature extraction i.e., without last fully-connected layer"""

    __Block_Instructors = {BlockType.BasicBlock: BasicBlock,
                           BlockType.BottleneckBlock: BottleneckBlock}

    def __init__(self, params: ResNetBackboneHyperParameterSet = ResNetBackboneHyperParameterSet()):
        if params.num_chanels is None:
            params.num_chanels = [64, 128, 256, 512]
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        # TODO: Originally the implementation outputted the layer-wise outputs as well
        #  -> Later it is needed for us
        in_channels = self.params.in_channels
        block_type = self.params.block_type
        num_blocks_per_layer = self.params.num_blocks_per_layer

        block_input_channels = 64
        block_class = self.__Block_Instructors[block_type]

        def _make_layer(num_channels: int, num_out_channels: int, num_blocks: int, stride: int):
            nonlocal block_input_channels
            nonlocal block_class

            strides = [stride] + [1] * (num_blocks - 1)
            num_out_channels = [num_channels] * (num_blocks - 1) + [num_out_channels]

            layers = []
            for num_out_channel, stride in zip(num_out_channels, strides):
                layers.append(block_class(block_input_channels, num_channels, num_out_channel, stride))
                block_input_channels = num_out_channel * block_class.expansion
            return nn.Sequential(*layers)

        if self.params.initial_kernel_reduced:
            self.conv1 = nn.Conv2d(in_channels, block_input_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, block_input_channels, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(block_input_channels)

        self.layer1 = _make_layer(self.params.num_chanels[0], self.params.num_chanels[0], num_blocks_per_layer[0],
                                  stride=1)
        self.layer2 = _make_layer(self.params.num_chanels[1], self.params.num_chanels[1], num_blocks_per_layer[1],
                                  stride=2)
        self.layer3 = _make_layer(self.params.num_chanels[2], self.params.num_chanels[2], num_blocks_per_layer[2],
                                  stride=2)
        self.layer4 = _make_layer(self.params.num_chanels[3], self.params.feature_size, num_blocks_per_layer[3],
                                  stride=2)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        return None

    def forward(self, x):
        intermediate_layers = {}

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        if "layer1" in self.params.intermediate_layers_to_return:
            intermediate_layers["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.params.intermediate_layers_to_return:
            intermediate_layers["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.params.intermediate_layers_to_return:
            intermediate_layers["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.params.intermediate_layers_to_return:
            intermediate_layers["layer4"] = x
        x = self.pooling(x)
        x = self.flatten(x)

        return x, intermediate_layers

    def initialize_model(self):
        pass

    @classmethod
    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        parser.add_argument('--block_type', type=BlockType, choices=list(BlockType))
        parser.add_argument('--num_blocks_per_layer', type=list)
        parser.add_argument('--in_channels', type=int, default=3)
        return parser


class ResNet18BackboneHyperParameterSet(ResNetBackboneHyperParameterSet):
    def __init__(self,
                 initial_kernel_reduced: bool = True,
                 feature_size: int = 512,
                 num_chanels: list = None,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 intermediate_layers_to_return: List[str] = [],
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(block_type=BlockType.BasicBlock,
                         num_blocks_per_layer=[2, 2, 2, 2],
                         in_channels=3,
                         feature_size=feature_size,
                         num_chanels=num_chanels,
                         initial_kernel_reduced=initial_kernel_reduced,
                         intermediate_layers_to_return=intermediate_layers_to_return,
                         optimizer_definition=optimizer_definition,
                         scheduler_definition=scheduler_definition,
                         loss_calc_params=loss_calc_params,
                         **kwargs)


class ResNet18BackboneDefinition(ResNetBackboneDefinition):
    def __init__(self, hyperparams: ResNet18BackboneHyperParameterSet = ResNet18BackboneHyperParameterSet()):
        super().__init__(ModelType.ResNet18Backbone, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet18Backbone


class ResNet18Backbone(ResNetBackbone):
    def __init__(self, params: ResNet18BackboneHyperParameterSet = ResNet18BackboneHyperParameterSet()):
        super().__init__(params)


class ResNet34BackboneHyperParameterSet(ResNetBackboneHyperParameterSet):
    def __init__(self,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 intermediate_layers_to_return: List[str] = [],
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(block_type=BlockType.BasicBlock,
                         num_blocks_per_layer=[3, 4, 6, 3],
                         in_channels=3,
                         initial_kernel_reduced=initial_kernel_reduced,
                         intermediate_layers_to_return=intermediate_layers_to_return,
                         feature_size=512,
                         num_chanels=[64, 128, 256, 512],
                         optimizer_definition=optimizer_definition,
                         scheduler_definition=scheduler_definition,
                         loss_calc_params=loss_calc_params,
                         **kwargs)


class ResNet34BackboneDefinition(ResNetBackboneDefinition):
    def __init__(self, hyperparams: ResNet34BackboneHyperParameterSet = ResNet34BackboneHyperParameterSet()):
        super().__init__(ModelType.ResNet34Backbone, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet34Backbone


class ResNet34Backbone(ResNetBackbone):
    def __init__(self, params: ResNet34BackboneHyperParameterSet = ResNet34BackboneHyperParameterSet()):
        super().__init__(params)


class ResNet50BackboneHyperParameterSet(ResNetBackboneHyperParameterSet):
    def __init__(self,
                 initial_kernel_reduced: bool = True,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 intermediate_layers_to_return: List[str] = [],
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs: Any):
        super().__init__(block_type=BlockType.BottleneckBlock,
                         num_blocks_per_layer=[3, 4, 6, 3],
                         in_channels=3,
                         initial_kernel_reduced=initial_kernel_reduced,
                         intermediate_layers_to_return=intermediate_layers_to_return,
                         feature_size=512,
                         num_chanels=[64, 128, 256, 512],
                         optimizer_definition=optimizer_definition,
                         scheduler_definition=scheduler_definition,
                         loss_calc_params=loss_calc_params,
                         **kwargs)


class ResNet50BackboneDefinition(ResNetBackboneDefinition):
    def __init__(self, hyperparams: ResNet50BackboneHyperParameterSet = ResNet50BackboneHyperParameterSet()):
        super().__init__(ModelType.ResNet50Backbone, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return ResNet50Backbone


class ResNet50Backbone(ResNetBackbone):
    def __init__(self, params: ResNet50BackboneHyperParameterSet = ResNet50BackboneHyperParameterSet()):
        super().__init__(params)
