"""
FixMatch based on https://github.com/kekmodel/FixMatch-pytorch
"""

import copy
from typing import Optional, Callable

import torch
from torch import nn
from tqdm.autonotebook import tqdm
import torch.nn.functional as F


from lowdataregime.active_learning.active_datamodule import ActiveDataModule
from lowdataregime.classification.data._transform.fixmatch import TransformFixMatch
from lowdataregime.classification.log.logger import init_logger
from lowdataregime.classification.model.classification_module import ClassificationModule
from lowdataregime.parameters.params import HyperParameterSet
from lowdataregime.classification.trainer.trainers import DeviceType, TrainerDefinition, TrainerType


class FixMatchTrainerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the FixMatchTrainer"""

    def __init__(self,
                 device: Optional[DeviceType] = DeviceType.GPU,
                 max_epochs=200,
                 fast_dev_run=False,
                 unlabeled_batch_size_coeff: int = 10,
                 psuedo_threshold: float = 0.95,
                 pseudo_temperature: float = 1.0,
                 unlabeled_loss_coeff: float = 1.0,
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param device: The device to be used
        :param max_epochs: The maximum number of epochs
        """
        super().__init__(**kwargs)
        self.device = device
        self.max_epochs = max_epochs
        self.fast_dev_run = fast_dev_run
        self.unlabeled_batch_size_coeff = unlabeled_batch_size_coeff
        self.psuedo_threshold = psuedo_threshold
        self.pseudo_temperature = pseudo_temperature
        self.unlabeled_loss_coeff = unlabeled_loss_coeff


class FixMatchTrainerDefinition(TrainerDefinition):
    """Definition of the FixMatchTrainer"""

    def __init__(self, hyperparams: FixMatchTrainerHyperParameterSet = FixMatchTrainerHyperParameterSet()):
        super().__init__(TrainerType.FixMatchTrainer, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return FixMatchTrainer


class FixMatchTrainer:

    def __init__(self, params: FixMatchTrainerHyperParameterSet = FixMatchTrainerHyperParameterSet(), **kwargs):
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

    def test(self, model, datamodule):
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

        return correct / total
        pass

    def _train(self, model: ClassificationModule, datamodule: ActiveDataModule):
        """
        Train the model on the data module
        :param model: The model to train
        :param datamodule: The data module on the model needs to be trained
        """
        self.logger.info('Train the model with FixMatch!')
        labeled_trainloader = datamodule.train_labeled_dataloader()

        dataset = datamodule.dataset.get_dataset(train=True, transform=TransformFixMatch(mean=datamodule.dataset.params.MEAN, std=datamodule.dataset.params.STD))
        unlabeled_trainloader = \
            datamodule.train_unlabeled_dataloader(dataset=dataset,
                                                  batch_size=datamodule.batch_size * self.params.unlabeled_batch_size_coeff)

        optimizer, scheduler = self._get_optimizer_scheduler(model)

        for epoch in tqdm(range(self.params.max_epochs), leave=False, total=self.params.max_epochs):
            if scheduler is not None:
                scheduler.step()

            self._train_epoch(epoch, model, optimizer, labeled_trainloader, unlabeled_trainloader)

        self.logger.info('Training finished!')

    def _get_optimizer_scheduler(self, model):
        optimization_variables = model.configure_optimizers()
        if isinstance(optimization_variables, tuple):
            optimizer = optimization_variables[0][0]
            scheduler = optimization_variables[1][0]
        else:
            optimizer = optimization_variables
            scheduler = None

        return optimizer, scheduler

    def _train_epoch(self, epoch, model, optimizer, labeled_trainloader, unlabeled_trainloader):
        """
        Train one epoch
        :param model: The model to be trained
        :param optimizer: The optimizer of the model
        :param train_loader: Data loader for the training
        """

        model.train()

        for (labeled_inputs, labeled_labels), ((unlabeled_weak_inputs, unlabeled_strong_inputs), _) in zip(labeled_trainloader, unlabeled_trainloader):
            optimizer.zero_grad()

            batch_size = labeled_inputs.shape[0]
            inputs = self.interleave(
                torch.cat((labeled_inputs, unlabeled_weak_inputs, unlabeled_strong_inputs)), 2*self.params.unlabeled_batch_size_coeff+1).to(self.params.device.value)
            targets_x = labeled_labels.to(self.params.device.value)
            logits = model(inputs)
            logits = self.de_interleave(logits, 2*self.params.unlabeled_batch_size_coeff+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach() / self.params.pseudo_temperature, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.params.psuedo_threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + self.params.unlabeled_loss_coeff * Lu

            loss.backward()

            optimizer.step()
            model.zero_grad()

    def interleave(self, x, size):
        """
        From https://github.com/kekmodel/FixMatch-pytorch
        """
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        """
        From https://github.com/kekmodel/FixMatch-pytorch
        """
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])