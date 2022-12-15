import copy
import os.path

import optuna
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional, Dict, Sequence
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F

from lowdataregime.classification.sampling.sampler import SubsetSequentialSampler, SamplerDefinition, \
    SubsetSequentialSamplerDefinition, SamplerType
from lowdataregime.classification.log.logger import init_logger
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace, DefinitionSpace
from lowdataregime.active_learning.query.queries import Query, QueryDefinition, QueryType
from lowdataregime.utils.utils import id_collate, IndexedDataset


class UncertaintyQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 inference_training_sample_ratio: float = 0.0,
                 temperature: float = 1,
                 unlabeled_pool_sampler_definition: SamplerDefinition = SubsetSequentialSamplerDefinition(),
                 log_query: bool = False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.inference_training_sample_ratio = inference_training_sample_ratio
        self.temperature = temperature
        self.unlabeled_pool_sampler_definition = unlabeled_pool_sampler_definition
        self.log_query = log_query

    def definition_space(self):
        return UncertaintyQueryHyperParameterSpace(self)


class UncertaintyQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: UncertaintyQueryHyperParameterSet = UncertaintyQueryHyperParameterSet()):
        super().__init__(QueryType.UncertaintyQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        raise NotImplementedError()

    def instantiate(self, log_folder, *args, **kwargs):
        """Instantiates the module"""
        return UncertaintyQuery(log_folder, self.hyperparams)

    def definition_space(self):
        return UncertaintyQueryDefinitionSpace(self.hyperparams.definition_space())


class UncertaintyQuery(Query):
    def __init__(self, log_folder: str,
                 params: UncertaintyQueryHyperParameterSet = UncertaintyQueryHyperParameterSet()):
        super().__init__(params)
        self.log_folder = log_folder
        self.logger = init_logger(self.__class__.__name__)

        self.inference_labeled_size = None
        self.inference_unlabeled_size = None

    @property
    def query_file(self):
        return os.path.join(self.log_folder, "uncertainty_query.csv")

    def sort_unlabeled_pool_by_metric(self, model, datamodule, num_samples_to_evaluate: int, ascending: bool = False):
        """
        Sorts the first <num_samples_to_evaluate> samples of the unlabeled pool by uncertainty
        :param num_samples_to_evaluate: The number of unlabeled samples to evaluate
        :return: Logits or uncertainties if requested
        """

        self.logger.info("Sorting the unlabeled pool by uncertainty!")
        uncertainties = self.evaluate_metric(model, datamodule, num_samples_to_evaluate)
        if ascending:
            uncertainties = -uncertainties
        query_logs = self.get_query_logs(datamodule, uncertainties)

        datamodule.sort_unlabeled_indices_by_list(uncertainties)

        self.logger.info("Unlabeled pool sorted by uncertainty!")
        return query_logs

    def get_query_logs(self, datamodule, uncertainties):
        samples_evaluated = len(uncertainties)
        query_logs = pd.DataFrame(
            {"sample_id": datamodule.unlabeled_pool_indices[:samples_evaluated],
             "entropy": uncertainties.tolist()},
        )

        query_logs = query_logs.sort_values("entropy", ascending=False)
        return query_logs

    def evaluate_metric(self, model, datamodule, num_samples_to_evaluate: int):
        """
        Estimate the uncertainty of the first <num_samples_to_evaluate> elements of the unlabeled pool
        :param num_samples_to_evaluate: The number of samples to evaluate from the unlabeled pool
        :return: Numpy array of uncertainties of the evaluated samples
        """
        self.logger.info("Evaluating uncertainty!")

        inference_labeled_size = int(
            datamodule.batch_size * self.params.inference_training_sample_ratio)
        inference_unlabeled_size = int(
            datamodule.batch_size * (1 - self.params.inference_training_sample_ratio))

        sampler_indices = datamodule.unlabeled_pool_indices[:num_samples_to_evaluate]
        num_samples_to_evaluate = len(sampler_indices)

        sampler = self._get_sampler(sampler_indices, model.feature_model, datamodule.unaugmented_dataset_train)

        sampler_indices_order = {sample_idx: idx for idx, sample_idx in
                                 zip(range(len(sampler_indices)), sampler_indices)}

        unlabeled_loader = DataLoader(IndexedDataset(datamodule.unaugmented_dataset_train),
                                      batch_size=inference_unlabeled_size,
                                      sampler=sampler,
                                      num_workers=datamodule.num_workers,
                                      drop_last=False,
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True,
                                      collate_fn=id_collate)

        if inference_labeled_size != 0:
            labeled_loader = DataLoader(
                datamodule.unaugmented_dataset_train, batch_size=inference_labeled_size,
                sampler=SubsetRandomSampler(datamodule.labeled_pool_indices),
                num_workers=datamodule.num_workers,
                drop_last=True,
                # more convenient if we maintain the order of subset
                pin_memory=True)
        else:
            labeled_loader = [(torch.tensor([], device=model.device),
                               torch.tensor([], device=model.device))]

        model.eval()
        uncertainty = torch.tensor([], device=model.device)
        evaluated_sample_indices = []

        self.logger.debug(
            f"Size of unlabelled loader: {len(unlabeled_loader)}. Size of labelled loader: {len(labeled_loader)}")

        # tqdm(unlabeled_loader, leave=False, total=len(unlabeled_loader), mininterval=10, desc="Evaluating")
        self.logger.info("Evaluating...")
        with torch.no_grad():
            for ids, (inputs, labels) in unlabeled_loader:
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                batch_uncertainty = torch.tensor([], device=model.device)
                scores_results = []

                for (labeled_inputs, _) in labeled_loader:
                    labeled_inputs = labeled_inputs.to(model.device)
                    compound_inputs = torch.cat((inputs, labeled_inputs), 0)

                    scores = model(compound_inputs)
                    scores = scores[:inference_unlabeled_size]
                    cool_scores = scores / self.params.temperature
                    probs = F.softmax(cool_scores, dim=1)
                    log_probs = torch.log(probs)
                    to_be_summed = probs * log_probs
                    entropy = torch.t(-torch.sum(to_be_summed, dim=1, keepdim=True))
                    batch_uncertainty = torch.cat((batch_uncertainty, entropy), 0)

                batch_uncertainty = torch.t(torch.mean(batch_uncertainty, dim=0, keepdim=True))
                uncertainty = torch.cat((uncertainty, batch_uncertainty), 0)
                evaluated_sample_indices.extend(ids)

        model.train()

        uncertainty = np.squeeze(uncertainty.cpu().numpy())
        evaluated_sample_indices = [sampler_indices_order[idx] for idx in evaluated_sample_indices]
        sorted_key_idxs = np.argsort(evaluated_sample_indices).tolist()
        uncertainty = uncertainty[sorted_key_idxs]

        return uncertainty

    def _get_sampler(self, indices, feature_model, dataset):
        if self.params.unlabeled_pool_sampler_definition.type == SamplerType.SubsetRandomSampler:
            self.params.unlabeled_pool_sampler_definition.hyperparams.indices = indices
        elif self.params.unlabeled_pool_sampler_definition.type == SamplerType.SubsetSequentialSampler:
            self.params.unlabeled_pool_sampler_definition.hyperparams.indices = indices
        elif self.params.unlabeled_pool_sampler_definition.type == SamplerType.CombineSampler:
            raise NotImplementedError("CombineSampler for querying requires pseudo labeling")
        elif self.params.unlabeled_pool_sampler_definition.type == SamplerType.ClassBasedSampler:
            raise NotImplementedError("ClassBasedSampler for querying requires pseudo labeling")
        elif self.params.unlabeled_pool_sampler_definition.type == SamplerType.RegionSampler:
            self.params.unlabeled_pool_sampler_definition.hyperparams.indices = indices
        else:
            raise NotImplementedError()

        self.params.unlabeled_pool_sampler_definition.hyperparams.indices = indices
        sampler = self.params.unlabeled_pool_sampler_definition.instantiate()

        if self.params.unlabeled_pool_sampler_definition.type == SamplerType.RegionSampler:
            sampler.feature_model = feature_model
            sampler.dataset = dataset

        return sampler


class UncertaintyQueryHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: UncertaintyQueryHyperParameterSet = UncertaintyQueryHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> UncertaintyQueryHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)

        if "temperature" in trial.user_attrs:
            hyperparam_set.temperature = trial.user_attrs["temperature"]
        else:
            hyperparam_set.temperature = 1.0

        return hyperparam_set


class UncertaintyQueryDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: UncertaintyQueryHyperParameterSpace = UncertaintyQueryHyperParameterSpace()):
        super().__init__(QueryType.UncertaintyQuery, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> UncertaintyQueryDefinition:
        return UncertaintyQueryDefinition(self.hyperparam_space.suggest(trial))
