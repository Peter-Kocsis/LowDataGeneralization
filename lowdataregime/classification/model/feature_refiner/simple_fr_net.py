from collections import OrderedDict
from collections import OrderedDict
from typing import Callable, Dict

import torch
import torch.nn as nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModule, \
    ClassificationModuleHyperParameterSet
from lowdataregime.classification.model.models import ModelType, ClassificationModuleDefinition
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition, AdamOptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition


class SimpleFRNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the MessagePassingNet"""

    def __init__(self,
                 backbone_size: int = 512,
                 feature_size: int = 64,
                 output_size: int = 10,
                 num_inner_layers: int = 1,
                 head_bias: bool = False,
                 optimizer_definition: OptimizerDefinition = AdamOptimizerDefinition(),
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param feature_size: The dimension of the latent space to reduce the 512 size input
        :param num_message_pass: The number of message passings
        :param num_heads: The number of heads in the multi-headed attentio
        :param dropout_prob: The probability of the Dropour layer
        :param output_size: The size of the output
        :func:`~ClassificationModuleHyperParameterSet.__init__`
        """
        super().__init__(optimizer_definition, scheduler_definition, loss_calc_params, **kwargs)
        self.backbone_size = backbone_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_inner_layers = num_inner_layers
        self.head_bias = head_bias


class SimpleFRNetDefinition(ClassificationModuleDefinition):
    """Definition of the HybridMessagePassingNet"""

    def __init__(self, hyperparams: SimpleFRNetHyperParameterSet = SimpleFRNetHyperParameterSet()):
        super().__init__(ModelType.SimpleFRNet, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return SimpleFRNet


class SimpleFRNet(ClassificationModule):
    """
    Message Passing Network for feature refinement, which includes the latent dimension reduction as first layer
    Inspiration: https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
    It uses TransformerConvLayers from PyTorch Geometric
    """

    def __init__(self, params: SimpleFRNetHyperParameterSet = SimpleFRNetHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        feature_size = self.params.feature_size

        # ====================== Dimension reduction ======================
        hybrid_fc = nn.Sequential(
            nn.Linear(self.params.backbone_size, feature_size),
            # nn.ReLU()
        )

        # ====================== Feature Refiner ======================
        layers = []

        # FR layers
        for idx in range(self.params.num_inner_layers):
            layers.append((f'linear_{idx}', nn.Linear(feature_size, feature_size)))
            layers.append((f'activation_{idx}', nn.ReLU()))

        fr_layers = nn.Sequential(OrderedDict(layers))

        # ====================== Final FC ======================
        fc = nn.Linear(feature_size, self.params.output_size, self.params.head_bias)

        # ====================== Whole module ======================
        return nn.ModuleDict(OrderedDict([
            ('hybrid_fc', hybrid_fc),
            ('fr', fr_layers),
            ('fc', fc)
        ]))

    def dimension_reduction(self, x):
        return self.model.hybrid_fc(x)

    def feature_refinement(self, x):
        return self.model.fr(x)

    def head(self, x):
        return self.model.fc(x)

    def forward(self, x, intermediate):
        """
        Runs the forward pass on the data
        :param data: Data to be forwarded
        :return: The output of the model
        """
        x = self.dimension_reduction(x)
        x = self.feature_refinement(x)
        x = self.head(x)
        return x

    def initialize_model(self):
        pass
