from collections import OrderedDict
from typing import Callable, Dict, Sequence, Any

import optuna
import torch
from torch import nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet, \
    ClassificationModule
from lowdataregime.classification.model.convolutional.resnet import ResNet18Definition
from lowdataregime.classification.model.feature_refiner.fr_net import FRNetHyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition


class FeatureRefinerHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the GeneralNet"""

    def __init__(self,
                 iid_net_def: ClassificationModuleDefinition = ResNet18Definition(),
                 fr_net_def: ClassificationModuleDefinition = FRNetHyperParameterSet(),
                 optimizer_definition: OptimizerDefinition = None,
                 scheduler_definition: SchedulerDefinition = None,
                 loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet] = {},
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param iid_net_def: The definition of the feature extraction part
        :param mpn_net_def: The definition of the message passing network part
        :func:`~ClassificationModuleHyperParameterSet.__init__`
        """
        super().__init__(optimizer_definition, scheduler_definition, loss_calc_params, **kwargs)
        self.iid_net_def = iid_net_def
        self.fr_net_def = fr_net_def


class FeatureRefinerDefinition(ClassificationModuleDefinition):
    """Definition of the GeneralNet"""

    def __init__(self, hyperparams: FeatureRefinerHyperParameterSet = FeatureRefinerHyperParameterSet()):
        super().__init__(ModelType.FeatureRefiner, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return FeatureRefiner


class FeatureRefiner(ClassificationModule):
    """
    GeneralNet
    """
    def __init__(self, params: FeatureRefinerHyperParameterSet = FeatureRefinerHyperParameterSet()):
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        iid_net = self.params.iid_net_def.instantiate()
        fr_net = self.params.fr_net_def.instantiate()

        return nn.Sequential(OrderedDict([
            ('iid_net', iid_net),
            ('fr_net', fr_net)
        ]))

    def initialize_model(self):
        pass

    def forward(self, x: torch.tensor):
        """
        Runs the forward pass on the data
        :param x: The input to be forwarded
        :return: The output of the model
        """
        x, intermediate = self.model.iid_net.features(x)
        self.model.iid_net.head(x)  # Run also the other head to have IID layer outputs
        x = self.model.fr_net(x, intermediate)
        return x

    def forward_with_intermediate(self, x: torch.tensor):
        """
        Runs the forward pass on the data
        :param x: The input to be forwarded
        :return: The output of the model
        """
        x, intermediate = self.model.iid_net.features(x)
        self.model.iid_net.head(x)  # Run also the other head to have IID layer outputs
        x = self.model.fr_net(x, intermediate)
        return x, intermediate

    def features(self, x):
        x, _ = self.model.iid_net.features(x)
        return x

    @property
    def feature_model(self):
        return self.model.iid_net.feature_model

    def forward_iid(self, x: torch.tensor):
        x = self.model.iid_net.forward(x)
        return x

    def forward_with_iid(self, x: torch.tensor):
        x = self.model.iid_net.features(x)
        iid_logits = self.model.iid_net.head(x)
        x = self.model.fr_net(x)
        return x, iid_logits
