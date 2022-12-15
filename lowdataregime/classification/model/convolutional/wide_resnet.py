"""
WideResNet, adapted from https://github.com/davda54/sam/blob/main/example/model/wide_res_net.py
"""

from collections import OrderedDict
from typing import Dict, Any, Callable

import torch.nn as nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model._network.gradient_gate import GradientGate
from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet, \
    ClassificationModule
from lowdataregime.classification.model.convolutional.wide_resnet_backbone import WideResNetResNetBackboneDefinition
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition


class WideResNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the ResNet"""

    def __init__(self,
                 backbone_definition: WideResNetResNetBackboneDefinition = WideResNetResNetBackboneDefinition(),
                 output_size: int = None,
                 gradient_multiplier: float = None,
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
        self.backbone_definition = backbone_definition
        self.output_size = output_size
        self.gradient_multiplier = gradient_multiplier
        self.feature_size = 512


class WideResNetDefinition(ClassificationModuleDefinition):
    def __init__(self, hyperparams: WideResNetHyperParameterSet = WideResNetHyperParameterSet()):
        super().__init__(ModelType.WideResNet, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return WideResNet

class WideResNet(ClassificationModule):
    def __init__(self, params: WideResNetHyperParameterSet = WideResNetHyperParameterSet()):
        super().__init__(params)

    def define_model(self):
        backbone = self.params.backbone_definition.instantiate()

        classifier = nn.Sequential(
            OrderedDict([
                ('gradient_gate', GradientGate(gradient_multiplier=self.params.gradient_multiplier)),
                ('classifier_layers',
                 nn.Linear(self.params.feature_size, self.params.output_size, bias=False))]))

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
        pass

