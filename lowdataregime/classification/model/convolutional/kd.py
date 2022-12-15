from collections import OrderedDict
from typing import Callable, Dict, Sequence, Any

import optuna
import torch
from torch import nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.loss.losses import CrossEntropyDefinition
from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet, \
    ClassificationModule, ClassificationModuleHyperParameterSpace, TrainStage
from lowdataregime.classification.model.convolutional.dml import DMLNet
from lowdataregime.classification.model.convolutional.resnet import ResNet18Definition, ResNet50Definition, \
    ResNet50HyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace
import pytorch_lightning as pl


class KDNetHyperParameterSet(ClassificationModuleHyperParameterSet):
    """HyperParameterSet of the GeneralNet"""

    def __init__(self,
                 main_net_def: ClassificationModuleDefinition = ResNet18Definition(),
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
        self.main_model_def = main_net_def

    def definition_space(self):
        return KDNetHyperParameterSpace(self)


class KDNetDefinition(ClassificationModuleDefinition):
    """Definition of the GeneralNet"""

    def __init__(self, hyperparams: KDNetHyperParameterSet = KDNetHyperParameterSet()):
        super().__init__(ModelType.KDNet, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return KDNet

    def definition_space(self):
        return KDNetDefinitionSpace(self.hyperparams.definition_space())


class KDNet(DMLNet):
    def define_model(self) -> torch.nn.Module:
        main_model = self.params.main_model_def.instantiate()

        teacher_loss_calc_params = {"cross_entropy": LossEvaluatorHyperParameterSet(
                                                        layers_needed=None,
                                                        loss_definition=CrossEntropyDefinition())}
        mutual_model_def = ResNet50Definition(
            ResNet50HyperParameterSet(
                output_size=self.params.main_model_def.hyperparams.output_size,
                head_bias=self.params.main_model_def.hyperparams.head_bias,
                gradient_multiplier=self.params.main_model_def.hyperparams.gradient_multiplier,
                optimizer_definition=self.params.main_model_def.hyperparams.optimizer_definition,
                scheduler_definition=self.params.main_model_def.hyperparams.scheduler_definition,
                loss_calc_params=teacher_loss_calc_params
            ))
        mutual_model = mutual_model_def.instantiate()

        return nn.ModuleDict(OrderedDict([
            ('main_model', main_model),
            ('mutual_model', mutual_model)
        ]))


class KDNetHyperParameterSpace(ClassificationModuleHyperParameterSpace):

    def __init__(self, default_hyperparam_set: KDNetHyperParameterSet = KDNetHyperParameterSet()):
        super().__init__(default_hyperparam_set)

    def suggest(self, trial: optuna.Trial) -> KDNetDefinition:
        raise NotImplementedError()


class KDNetDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: KDNetHyperParameterSpace = KDNetHyperParameterSpace()):
        super().__init__(ModelType.GeneralNet, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> KDNetDefinition:
        return KDNetDefinition(self.hyperparam_space.suggest(trial))
