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


class WideResNetBackboneHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the ResNet"""

    def __init__(self,
                 depth: int = 16,
                 width_factor: int = 8,
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
        self.depth = depth
        self.width_factor = width_factor
        self.head_bias = head_bias

        self.in_channels = 3
        self.dropout = 0.
        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)


class WideResNetResNetBackboneDefinition(ClassificationModuleDefinition):
    def __init__(self, hyperparams: WideResNetBackboneHyperParameterSet = WideResNetBackboneHyperParameterSet()):
        super().__init__(ModelType.WideResNetBackbone, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return WideResNetBackbone


class WideResNetBackbone(ClassificationModule):
    """WideResNet-based architecture only for feature extraction i.e., without last fully-connected layer"""

    def __init__(self, params: WideResNetBackboneHyperParameterSet = WideResNetBackboneHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        model = nn.Sequential(OrderedDict([
            ("0_convolution",
             nn.Conv2d(self.params.in_channels, self.params.filters[0], (3, 3), stride=1, padding=1, bias=False)),
            ("1_block", Block(self.params.filters[0], self.params.filters[1], 1, self.params.block_depth, self.params.dropout)),
            ("2_block", Block(self.params.filters[1], self.params.filters[2], 2, self.params.block_depth, self.params.dropout)),
            ("3_block", Block(self.params.filters[2], self.params.filters[3], 2, self.params.block_depth, self.params.dropout)),
            ("4_normalization", nn.BatchNorm2d(self.params.filters[3])),
            ("5_activation", nn.ReLU(inplace=True)),
            ("6_pooling", nn.AdaptiveAvgPool2d((1, 1))),
            ("7_flattening", nn.Flatten()),
        ]))
        return model

    def forward(self, x):
        intermediate_layers = {}

        x = self.model(x)

        return x, intermediate_layers

    def initialize_model(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.zero_()
        #         m.bias.data.zero_()

    @classmethod
    def add_argparse_args(cls, parent_parser):
        super_parser = ClassificationModule.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser], add_help=False)
        return parser


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int,
                 dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(in_channels)),
            ("1_activation", nn.ReLU(inplace=True)),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution",
             nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
            ("1_normalization", nn.BatchNorm2d(out_channels)),
            ("2_activation", nn.ReLU(inplace=True)),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution",
             nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))
        self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0,
                                    bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(channels)),
            ("1_activation", nn.ReLU(inplace=True)),
            ("2_convolution",
             nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
            ("3_normalization", nn.BatchNorm2d(channels)),
            ("4_activation", nn.ReLU(inplace=True)),
            ("5_dropout", nn.Dropout(dropout, inplace=True)),
            ("6_convolution",
             nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))

    def forward(self, x):
        return self.block(x)
