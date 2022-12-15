
import os
from typing import Callable

from lowdataregime.active_learning.active_trainer import ActiveTrainer
from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet
from lowdataregime.active_learning.active_datamodule import ActiveDataModuleStatus, ActiveDataModuleDefinition
from lowdataregime.utils.utils import Serializable


class ActiveLoaderHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 experiment_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.experiment_path = experiment_path


class ActiveLoaderDefinition(DefinitionSet):
    """Definition of the ActiveTrainer"""

    def __init__(self,
                 hyperparams: ActiveLoaderHyperParameterSet = ActiveLoaderHyperParameterSet()):
        super().__init__(hyperparams=hyperparams)

    def instantiate(self, *args, **kwargs):
        """Instantiates the module"""
        return ActiveLoader(params=self.hyperparams)


class ActiveLoader:
    MODEL_CHECKPOINT_NAME = "last.ckpt"
    DATA_CHECKPOINT_NAME = "data.json"

    def __init__(self, params: ActiveLoaderHyperParameterSet):
        self.params = params
        self.experiment_path = params.experiment_path

    @staticmethod
    def _get_stage_id(current_stage: int):
        return f"stage_{current_stage}"

    def _get_stage_folder_path(self, stage):
        return os.path.join(self.experiment_path, self._get_stage_id(stage))

    def _get_checkpoint(self, stage: int, checkpoint_name: str):
        return os.path.join(self.experiment_path, self._get_stage_id(stage), checkpoint_name)

    def _get_training_params(self):
        return Serializable.loads_from_file(os.path.join(self.experiment_path, ActiveTrainer.PARAMS_FILE))

    def load_active_datamodule(self, stage):
        active_datamodule_status = ActiveDataModuleStatus.loads_from_file(
            self._get_checkpoint(stage, self.DATA_CHECKPOINT_NAME))
        training_params = self._get_training_params()
        active_datamodule_params = training_params.datamodule_hyperparams
        active_datamodule_definition = ActiveDataModuleDefinition(active_datamodule_params, active_datamodule_status)
        datamodule_definition = training_params.optimization_definition_set.data_definition
        return active_datamodule_definition.instantiate(datamodule_definition)

    def load_model(self, stage):
        training_params = self._get_training_params()
        the_model = training_params.optimization_definition_set.model_definition.instantiate()
        model_checkpoint = self._get_checkpoint(stage, self.MODEL_CHECKPOINT_NAME)
        if os.path.exists(model_checkpoint):
            the_model = the_model.load_from_checkpoint(model_checkpoint)
        return the_model


if __name__ == "__main__":
    active_loader = ActiveLoader(ActiveLoaderHyperParameterSet(experiment_path='logs/active_learning/dummy/cifar10/'
                                                                               'training_-1_20210403-141233'))
    datamodule = active_loader.load_active_datamodule(stage=3)
    model = active_loader.load_model(stage=3)
    print(model.params)

