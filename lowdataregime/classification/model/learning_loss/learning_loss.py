from collections import OrderedDict
from typing import Callable, Dict, Sequence, Any

import optuna
import torch
from torch import nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet, \
    ClassificationModule, ClassificationModuleHyperParameterSpace
from lowdataregime.classification.model.convolutional.resnet import ResNet18Definition
from lowdataregime.classification.model.learning_loss.lossnet import LossNetDefinition
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace


class LearningLossHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the LearningLoss"""

    def __init__(self,
                 loss_net_def: ClassificationModuleDefinition = LossNetDefinition(),
                 main_model_def: ClassificationModuleDefinition = ResNet18Definition(),
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
        self.loss_net_def = loss_net_def
        self.main_model_def = main_model_def

    def definition_space(self):
        return LearningLossHyperParameterSpace(self)


class LearningLossDefinition(ClassificationModuleDefinition):
    """Definition of the LearningLoss"""

    def __init__(self, hyperparams: LearningLossHyperParameterSet = LearningLossHyperParameterSet()):
        super().__init__(ModelType.LearningLoss, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return LearningLoss

    def definition_space(self):
        return LearningLossDefinitionSpace(self.hyperparams.definition_space())


class LearningLoss(ClassificationModule):
    """
    LearningLoss
    """
    def __init__(self, params: LearningLossHyperParameterSet = LearningLossHyperParameterSet()):
        self.edge_index = None
        super().__init__(params)

    def define_model(self) -> torch.nn.Module:
        loss_net = self.params.loss_net_def.instantiate()
        main_model = self.params.main_model_def.instantiate()

        return nn.ModuleDict(OrderedDict([
            ('loss_net', loss_net),
            ('main_model', main_model)
        ]))

    def initialize_model(self):
        pass

    def forward(self, x: torch.tensor):
        """
        Runs the forward pass on the data
        :param x: The input to be forwarded
        :return: The output of the model
        """
        scores, features = self.model.main_model.forward_with_intermediate(x)
        pred_loss = self.model.loss_net(list(features.values()))
        return pred_loss

    @property
    def feature_model(self):
        return None


class LearningLossHyperParameterSpace(ClassificationModuleHyperParameterSpace):
    """HyperParameterSpace of the LearningLoss"""

    def __init__(self,
                 default_hyperparam_set: LearningLossHyperParameterSet = LearningLossHyperParameterSet(),
                 ):
        super().__init__(default_hyperparam_set)

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        return {}

    def suggest(self, trial: optuna.Trial) -> LearningLossHyperParameterSet:
        """
        Sugges new HyperParameterSet for a trial
        :return: Suggested HyperParameterSet
        """
        hyperparams = super().suggest(trial=trial)
        return hyperparams


class LearningLossDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the LearningLoss"""

    def __init__(self, hyperparam_space: LearningLossHyperParameterSpace = LearningLossHyperParameterSpace()):
        super().__init__(ModelType.LearningLoss, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> LearningLossDefinition:
        return LearningLossDefinition(self.hyperparam_space.suggest(trial))
