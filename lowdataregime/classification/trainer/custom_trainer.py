from typing import Optional, Callable

import torch
from tqdm.autonotebook import tqdm

from lowdataregime.classification.log.logger import init_logger
from lowdataregime.classification.model.classification_module import ClassificationModule
from lowdataregime.parameters.params import HyperParameterSet
from lowdataregime.classification.trainer.trainers import DeviceType, TrainerDefinition, TrainerType


class CustomTrainerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the CustomTrainer"""

    def __init__(self, device: Optional[DeviceType] = DeviceType.GPU,
                 max_epochs=30, **kwargs):
        """
        Creates new HyperParameterSet
        :param device: The device to be used
        :param max_epochs: The maximum number of epochs
        """
        super().__init__(**kwargs)
        self.device = device
        self.max_epochs = max_epochs


class CustomTrainerDefinition(TrainerDefinition):
    """Definition of the CustomTrainer"""

    def __init__(self, hyperparams: CustomTrainerHyperParameterSet = CustomTrainerHyperParameterSet()):
        super().__init__(TrainerType.CustomTrainer, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return CustomTrainer


class CustomTrainer:

    def __init__(self, params: CustomTrainerHyperParameterSet = CustomTrainerHyperParameterSet(), **kwargs):
        self.params = params
        self.logger = init_logger(self.__class__.__name__)

    def fit(self, model: ClassificationModule, datamodule):
        """
        Fit model to the data module
        :param model: The model to fit
        :param datamodule: The data module to fit the model
        """
        model = model.to(self.params.device.value)

        self._train(model, datamodule)

    def test(self, model: ClassificationModule, datamodule):
        """
        Test the model on the datamodule
        """
        model = model.to(self.params.device.value)

        model.eval()
        test_loader = datamodule.test_dataloader()

        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs, labels) in tqdm(test_loader, leave=False, total=len(test_loader)):
                inputs = inputs.to(self.params.device.value)
                labels = labels.to(self.params.device.value)

                scores = model(inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return 100 * correct / total

    def _train(self, model: ClassificationModule, datamodule):
        """
        Train the model on the data module
        :param model: The model to train
        :param datamodule: The data module on the model needs to be trained
        """
        self.logger.info('Train the model!')
        train_loader = datamodule.train_dataloader()

        optimization_variables = model.configure_optimizers()
        if isinstance(optimization_variables, tuple):
            optimizer = optimization_variables[0][0]
            scheduler = optimization_variables[1][0]
        else:
            optimizer = optimization_variables
            scheduler = None

        for epoch in tqdm(range(self.params.max_epochs), leave=False, total=self.params.max_epochs):
            if scheduler is not None:
                scheduler.step()

            self._train_epoch(model, optimizer, train_loader)

        self.logger.info('Training finished!')

    def _train_epoch(self, model: ClassificationModule, optimizer, train_loader):
        """
        Train one epoch
        :param model: The model to be trained
        :param optimizer: The optimizer of the model
        :param train_loader: Data loader for the training
        """
        model.train()

        for batch_idx, batch in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
            batch = (batch[0].to(self.params.device.value), batch[1].to(self.params.device.value))

            optimizer.zero_grad()

            loss = model.training_step(batch, batch_idx)

            loss.backward()
            optimizer.step()