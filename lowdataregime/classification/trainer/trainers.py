import copy
from abc import ABC
from typing import Callable, Optional, Any, Sequence, Dict

import optuna
import torch
from pytorch_lightning import Trainer

from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet, HyperParameterSpace, DefinitionSpace
from lowdataregime.utils.utils import SerializableEnum


class DeviceType(SerializableEnum):
    """Definition of the available devices"""
    CPU = "cpu"
    GPU = "cuda"


class TrainerType(SerializableEnum):
    """Definition of the available trainers"""
    PL_Trainer = "PL_Trainer"
    LearningLossTrainer = "LearningLossTrainer"
    FixMatchTrainer = "FixMatchTrainer"


class TrainerDefinition(DefinitionSet, ABC):
    """Abstract definition of a Trainer"""

    def __init__(self, type: TrainerType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)


# ----------------------------------- PL_Trainer -----------------------------------


class PL_TrainerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the PyTorchLightningTrainer"""

    def __init__(self,
                 runtime_mode: Optional[DeviceType] = None,
                 max_epochs=30,
                 min_epochs=1,
                 weights_summary='full',
                 progress_bar_refresh_rate=20,
                 track_grad_norm=-1,
                 fast_dev_run=False,
                 overfit_batches=0.0,
                 checkpoint_callback=False,
                 check_val_every_n_epoch=1,
                 deterministic=False,
                 **kwargs: Any):
        """
        Creates new HyperParameterSet
        :param runtime_mode: The device to be used
        :func:`~Trainer.__init__`
        """
        super().__init__(**kwargs)

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.weights_summary = weights_summary
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.track_grad_norm = track_grad_norm
        self.fast_dev_run = fast_dev_run
        self.overfit_batches = overfit_batches
        self.checkpoint_callback = checkpoint_callback
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.deterministic = deterministic

        self.gpus = None

        if runtime_mode is None:
            runtime_mode = DeviceType.CPU

        if runtime_mode == DeviceType.GPU:
            self.gpus = -1

    @property
    def device(self):
        if self.gpus is None:
            return 'cpu'
        else:
            return 'cuda'

    def definition_space(self):
        return PL_TrainerHyperParameterSpace(self)


class PL_TrainerDefinition(TrainerDefinition):
    """Definition of the PyTorchLightningTrainer"""

    def __init__(self, hyperparams: PL_TrainerHyperParameterSet = PL_TrainerHyperParameterSet()):
        super().__init__(TrainerType.PL_Trainer, hyperparams)

    def instantiate(self, *args, **kwargs):
        return Trainer(*args, **self.hyperparams, **kwargs)

    def definition_space(self):
        return PL_TrainerDefinitionSpace(self.hyperparams.definition_space())


class PL_TrainerHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: PL_TrainerHyperParameterSet = PL_TrainerHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> PL_TrainerHyperParameterSet:
        hyperparams = copy.deepcopy(self.default_hyperparam_set)

        if hyperparams.max_epochs is None:
            if "num_epochs" in trial.user_attrs:
                hyperparams.max_epochs = trial.user_attrs["num_epochs"]
                hyperparams.min_epochs = trial.user_attrs["num_epochs"]
            else:
                hyperparams.max_epochs = 200
                hyperparams.min_epochs = 30
        return hyperparams


class PL_TrainerDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: PL_TrainerHyperParameterSpace = PL_TrainerHyperParameterSpace()):
        super().__init__(TrainerType.PL_Trainer, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> PL_TrainerDefinition:
        return PL_TrainerDefinition(self.hyperparam_space.suggest(trial))
