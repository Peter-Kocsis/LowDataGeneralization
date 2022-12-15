from typing import Optional, Callable

import torch
from torch import nn
from tqdm.autonotebook import tqdm

from lowdataregime.classification.log.logger import init_logger
from lowdataregime.classification.model.classification_module import ClassificationModule
from lowdataregime.classification.model.learning_loss.learning_loss import LearningLoss
from lowdataregime.parameters.params import HyperParameterSet
from lowdataregime.classification.trainer.trainers import DeviceType, TrainerDefinition, TrainerType


class LearningLossTrainerHyperParameterSet(HyperParameterSet):
    """HyperParameterSet of the LearningLossTrainer"""

    def __init__(self,
                 device: Optional[DeviceType] = DeviceType.GPU,
                 max_epochs=200,
                 lossnet_sheduling=120,
                 lossnet_weight=1.0,
                 lossnet_margin=1.0,
                 fast_dev_run=False,
                 **kwargs):
        """
        Creates new HyperParameterSet
        :param device: The device to be used
        :param max_epochs: The maximum number of epochs
        """
        super().__init__(**kwargs)
        self.device = device
        self.max_epochs = max_epochs
        self.lossnet_sheduling = lossnet_sheduling
        self.lossnet_weight = lossnet_weight
        self.lossnet_margin = lossnet_margin
        self.fast_dev_run = fast_dev_run


class LearningLossTrainerDefinition(TrainerDefinition):
    """Definition of the LearningLossTrainer"""

    def __init__(self, hyperparams: LearningLossTrainerHyperParameterSet = LearningLossTrainerHyperParameterSet()):
        super().__init__(TrainerType.LearningLossTrainer, hyperparams)

    @property
    def _instantiate_func(self) -> Callable:
        return LearningLossTrainer


class LearningLossTrainer:

    def __init__(self, params: LearningLossTrainerHyperParameterSet = LearningLossTrainerHyperParameterSet(), **kwargs):
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

    def test(self, model: LearningLoss, datamodule):
        """
        Test the model on the datamodule
        """
        assert isinstance(model, LearningLoss), f"LearningLoss Trainer accepts only LearningLoss model, not {model}!"
        model = model.to(self.params.device.value)

        model.eval()
        test_loader = datamodule.test_dataloader()

        main_model = model.model.main_model

        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs, labels) in tqdm(test_loader, leave=False, total=len(test_loader)):
                inputs = inputs.to(self.params.device.value)
                labels = labels.to(self.params.device.value)

                scores = main_model(inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return correct / total

    def _train(self, model: ClassificationModule, datamodule):
        """
        Train the model on the data module
        :param model: The model to train
        :param datamodule: The data module on the model needs to be trained
        """
        assert isinstance(model, LearningLoss), f"LearningLoss Trainer accepts only LearningLoss model, not {model}!"

        self.logger.info('Train the model!')
        train_loader = datamodule.train_dataloader()

        models = {}
        models["loss_net"] = model.model.loss_net
        models["main_model"] = model.model.main_model

        optimizers, schedulers = {}, {}
        optimizers["loss_net"], schedulers["loss_net"] = self._get_optimizer_scheduler(models["loss_net"])
        optimizers["main_model"], schedulers["main_model"] = self._get_optimizer_scheduler(models["main_model"])

        for epoch in tqdm(range(self.params.max_epochs), leave=False, total=self.params.max_epochs):
            for scheduler in schedulers.values():
                if scheduler is not None:
                    scheduler.step()

            self._train_epoch(epoch, models, optimizers, train_loader)

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

    def _train_epoch(self, epoch, models, optimizers, train_loader):
        """
        Train one epoch
        :param model: The model to be trained
        :param optimizer: The optimizer of the model
        :param train_loader: Data loader for the training
        """
        criterion = nn.CrossEntropyLoss(reduction='none')

        models["loss_net"].train()
        models["main_model"].train()

        # for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
        for inputs, labels in train_loader:
            inputs, labels = (inputs.to(self.params.device.value), labels.to(self.params.device.value))

            for optimizer in optimizers.values():
                optimizer.zero_grad()

            scores, features = models['main_model'].forward_with_intermediate(inputs)
            target_loss = criterion(scores, labels)

            if epoch > self.params.lossnet_sheduling:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                for feature_id in features.keys():
                    features[feature_id] = features[feature_id].detach()
            pred_loss = models['loss_net'](list(features.values()))
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = self.LossPredLoss(pred_loss, target_loss, margin=self.params.lossnet_margin)
            loss = m_backbone_loss + self.params.lossnet_weight * m_module_loss

            loss.backward()
            optimizers['main_model'].step()
            optimizers['loss_net'].step()

    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[
                :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss