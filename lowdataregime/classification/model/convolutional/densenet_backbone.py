from argparse import ArgumentParser
from collections import OrderedDict
from typing import List, Any, Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock, _Transition

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model._network.resnet_block import BasicBlock, BottleneckBlock, BlockType
from lowdataregime.classification.model.classification_module import ClassificationModule, \
    ClassificationModuleHyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition


class DenseNetBackboneHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the ResNet"""

    def __init__(self,
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
        self.num_init_features = 64
        self.block_config = (6, 12, 24, 16)
        self.bn_size = 4
        self.growth_rate = 32


class DenseNetResNetBackboneDefinition(ClassificationModuleDefinition):
    def __init__(self, hyperparams: DenseNetBackboneHyperParameterSet = DenseNetBackboneHyperParameterSet()):
        super().__init__(ModelType.DenseNetBackbone, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return DenseNetBackbone


class DenseNetBackbone(ClassificationModule):
    r"""
    Based on the official PyTorch implementation
    Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, params: DenseNetBackboneHyperParameterSet = DenseNetBackboneHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        # First convolution
        features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, self.params.num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(self.params.num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = self.params.num_init_features
        for i, num_layers in enumerate(self.params.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.params.bn_size,
                growth_rate=self.params.growth_rate,
                drop_rate=0,
                memory_efficient=False,
            )
            features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * self.params.growth_rate
            if i != len(self.params.block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        features.add_module("norm5", nn.BatchNorm2d(num_features))

        model = nn.Sequential(OrderedDict([
            ("0_features", features),
            ("1_relu", nn.ReLU(inplace=True)),
            ("2_pooling", nn.AdaptiveAvgPool2d((1, 1))),
            ("3_flattening", nn.Flatten()),
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
