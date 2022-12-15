from typing import Tuple, List, Optional

import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback

from lowdataregime.classification.data.classification_datamodule import ClassificationDataModule
from lowdataregime.classification.log.logger import TENSORBOARD_LOG_DIR, get_timestamp, ClassificationLogger
from lowdataregime.classification.model.classification_module import ClassificationModule
from lowdataregime.parameters.params import OptimizationDefinitionSet
from lowdataregime.utils.summary import summary


def fit(
        optimization_definition_set: OptimizationDefinitionSet,
        model: ClassificationModule,
        datamodule: ClassificationDataModule,
        save_dir: Optional[str] = TENSORBOARD_LOG_DIR,
        model_name: Optional[str] = "default",
        version: str = get_timestamp(),
        callbacks: Optional[List[Callback]] = None) -> Trainer:
    """
    Fits model
    :param optimization_definition_set: The definition of the optimization
    :param model: The model to fit
    :param datamodule: The data module to fit the model
    :param save_dir: The root directory of the logs
    :param model_name: The name of the model
    :param version: The version of the model
    :param callbacks: The PyTorchLightning callbacks of the training
    :return: The trainer
    """
    # -----------------------------------------------
    # Initialize the environment
    # ------------------------------------------------
    pl.seed_everything(optimization_definition_set.seed)

    # -----------------------------------------------
    # Logging
    # -----------------------------------------------
    logger = ClassificationLogger(save_dir=save_dir,
                                  name=model_name,
                                  version=version)
    logger.log_parameters(optimization_definition_set)

    # -----------------------------------------------
    # Training
    # -----------------------------------------------
    trainer: Trainer = optimization_definition_set.trainer_definition.instantiate(logger=logger,
                                                                                  callbacks=callbacks)

    summary(model.to(optimization_definition_set.trainer_definition.hyperparams.device),
            datamodule.dims,
            batch_size=datamodule.batch_size,
            device=optimization_definition_set.trainer_definition.hyperparams.device)
    trainer.fit(model=model, datamodule=datamodule)

    return trainer


def train_model(
        optimization_definition_set: OptimizationDefinitionSet,
        callbacks: Optional[List[Callback]] = None,
        save_dir: str = TENSORBOARD_LOG_DIR,
        model_name: str = "default",
        version: str = get_timestamp()) \
        -> Tuple[Trainer, ClassificationModule, ClassificationDataModule]:
    """
    Train a model of the given type with the given optimization_parameter_set on the given datamodule_type
    :param optimization_definition_set: The definition of the optimization
    :param callbacks: The PyTorchLightning callbacks of the training
    :param model_name: The name of the model
    :param version: The version of the model
    :return: Tuple of the used Trainer, the trained model, and the Datamodule
    """
    # -----------------------------------------------
    # Data
    # -----------------------------------------------
    datamodule = optimization_definition_set.data_definition.instantiate()

    # -----------------------------------------------
    # Model
    # -----------------------------------------------
    model = optimization_definition_set.model_definition.instantiate()

    # -----------------------------------------------
    # Training
    # -----------------------------------------------
    trainer = fit(optimization_definition_set=optimization_definition_set,
                  model=model,
                  datamodule=datamodule,
                  save_dir=save_dir, model_name=model_name, version=version,
                  callbacks=callbacks)

    return trainer, model, datamodule
