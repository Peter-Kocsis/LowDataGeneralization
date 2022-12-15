from collections import OrderedDict
from typing import Callable, Dict, Sequence, Any

import optuna
import torch
from torch import nn

from lowdataregime.classification.loss.loss_calculator import LossEvaluatorHyperParameterSet
from lowdataregime.classification.model.classification_module import ClassificationModuleHyperParameterSet, \
    ClassificationModule, ClassificationModuleHyperParameterSpace, TrainStage
from lowdataregime.classification.model.convolutional.resnet import ResNet18Definition, ResNet50Definition, \
    ResNet50HyperParameterSet
from lowdataregime.classification.model.models import ClassificationModuleDefinition, ModelType
from lowdataregime.classification.optimizer.optimizers import OptimizerDefinition
from lowdataregime.classification.optimizer.schedulers import SchedulerDefinition
from lowdataregime.parameters.params import DefinitionSpace
import pytorch_lightning as pl


class DMLNetHyperParameterSet(ClassificationModuleHyperParameterSet):
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
        return DMLNetHyperParameterSpace(self)


class DMLNetDefinition(ClassificationModuleDefinition):
    """Definition of the GeneralNet"""

    def __init__(self, hyperparams: DMLNetHyperParameterSet = DMLNetHyperParameterSet()):
        super().__init__(ModelType.DMLNet, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return DMLNet

    def definition_space(self):
        return DMLNetDefinitionSpace(self.hyperparams.definition_space())


class DMLNet(ClassificationModule):
    """
    GeneralNet
    """
    def __init__(self, params: DMLNetHyperParameterSet = DMLNetHyperParameterSet()):
        super().__init__(params)
        self.train_acc = None
        self.valid_acc = None
        self.test_mutual_acc = None

    def define_model(self) -> torch.nn.Module:
        main_model = self.params.main_model_def.instantiate()
        mutual_model_def = ResNet50Definition(
                    ResNet50HyperParameterSet(
                        output_size=self.params.main_model_def.hyperparams.output_size,
                        head_bias=self.params.main_model_def.hyperparams.head_bias,
                        gradient_multiplier=self.params.main_model_def.hyperparams.gradient_multiplier,
                        optimizer_definition=self.params.main_model_def.hyperparams.optimizer_definition,
                        scheduler_definition=self.params.main_model_def.hyperparams.scheduler_definition,
                        loss_calc_params=self.params.main_model_def.hyperparams.loss_calc_params
                    ))
        mutual_model = mutual_model_def.instantiate()

        return nn.ModuleDict(OrderedDict([
            ('main_model', main_model),
            ('mutual_model', mutual_model)
        ]))

    def initialize_model(self):
        pass

    @property
    def feature_model(self):
        return None

    def forward(self, x: torch.tensor):
        """
        Runs the forward pass on the data
        :param x: The input to be forwarded
        :return: The output of the model
        """
        x_main = self.model.main_model(x)
        x_mutual = self.model.mutual_model(x)
        return x_main, x_mutual

    def init_metrics(self):
        """Initializes the metric"""
        self.train_mutual_acc = pl.metrics.Accuracy().to(self.device)
        self.valid_mutual_acc = pl.metrics.Accuracy(compute_on_step=False).to(self.device)
        self.test_mutual_acc = pl.metrics.Accuracy(compute_on_step=False).to(self.device)
        super().init_metrics()

    def get_accuracy_metric(self, mode: TrainStage):
        """
        Returns the accuracy metric of the model
        :param mode: The TrainStage, whose accuracy metric is requested
        :return: The requested accuracy metric
        """
        if mode == TrainStage.Training:
            return self.train_acc, self.train_mutual_acc
        if mode == TrainStage.Validation:
            return self.valid_acc, self.valid_mutual_acc
        if mode == TrainStage.Test:
            return self.test_acc, self.test_mutual_acc
        raise RuntimeError(f"Unknown stage: {mode}")

    def general_step(self, batch, batch_idx, mode: TrainStage):
        """
        General step used in all phases
        :param batch: The current batch of data
        :param batch_idx: The current batch index
        :param mode: The current phase
        :return: The loss
        """
        x, y = batch

        # register loss inspection
        self.loss_calculator.inspect_layers(self)

        # forward pass
        logits = self(x)

        # loss
        loss, sub_losses = self.loss_calculator(logits, y)

        # deregister loss inspection
        self.loss_calculator.remove_inspection()

        # accuracy
        accuracy_metric, accuracy_metric_mutual = self.get_accuracy_metric(mode)
        accuracy_metric(logits[0], y)
        accuracy_metric_mutual(logits[1], y)

        if mode != TrainStage.Test:
            self.log(f"{mode}_loss", loss, logger=True)
            self.log(f"{mode}_acc", accuracy_metric, logger=True)
            self.log(f"{mode}_acc_mutual", accuracy_metric_mutual, logger=True)
            # self.log(f"{mode}_logits", logits, logger=False, on_epoch=True, reduce_fx=lambda x: x)
            # self.log(f"{mode}_targets", y, logger=False, on_epoch=True, reduce_fx=lambda x: x)

        for key, sub_loss in sub_losses.items():
            self.log(f"{mode}_loss_{key}", sub_loss, logger=True)

        return loss


class DMLNetHyperParameterSpace(ClassificationModuleHyperParameterSpace):

    def __init__(self, default_hyperparam_set: DMLNetHyperParameterSet = DMLNetHyperParameterSet()):
        super().__init__(default_hyperparam_set)

    def suggest(self, trial: optuna.Trial) -> DMLNetDefinition:
        raise NotImplementedError()


class DMLNetDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: DMLNetHyperParameterSpace = DMLNetHyperParameterSpace()):
        super().__init__(ModelType.GeneralNet, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> DMLNetDefinition:
        return DMLNetDefinition(self.hyperparam_space.suggest(trial))
