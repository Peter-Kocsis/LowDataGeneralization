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


class ModelQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 unlabeled_pool_sampler_definition: SamplerDefinition = SubsetSequentialSamplerDefinition(),
                 log_query: bool = False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.unlabeled_pool_sampler_definition = unlabeled_pool_sampler_definition
        self.log_query = log_query

    def definition_space(self):
        return ModelQueryHyperParameterSpace(self)


class ModelQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: ModelQueryHyperParameterSet = ModelQueryHyperParameterSet()):
        super().__init__(QueryType.ModelQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        raise NotImplementedError()

    def instantiate(self, log_folder, *args, **kwargs):
        """Instantiates the module"""
        return ModelQuery(log_folder, self.hyperparams)

    def definition_space(self):
        return ModelQueryDefinitionSpace(self.hyperparams.definition_space())


class ModelQuery(Query):
    def __init__(self, log_folder: str,
                 params: ModelQueryHyperParameterSet = ModelQueryHyperParameterSet()):
        super().__init__(params)
        self.log_folder = log_folder
        self.logger = init_logger(self.__class__.__name__)

    @property
    def query_file(self):
        return os.path.join(self.log_folder, "model_query.csv")

    def sort_unlabeled_pool_by_metric(self, model, datamodule, num_samples_to_evaluate: int, ascending: bool = False):
        """
        Sorts the first <num_samples_to_evaluate> samples of the unlabeled pool by uncertainty
        :param num_samples_to_evaluate: The number of unlabeled samples to evaluate
        :return: Logits or uncertainties if requested
        """
        self.logger.info("Sorting the unlabeled pool by uncertainty!")
        metric = self.evaluate_metric(model, datamodule, num_samples_to_evaluate)
        if ascending:
            metric = -metric
        query_logs = self.get_query_logs(datamodule, metric)

        datamodule.sort_unlabeled_indices_by_list(metric)

        self.logger.info("Unlabeled pool sorted by uncertainty!")
        return query_logs

    def get_query_logs(self, datamodule, metric):
        samples_evaluated = len(metric)
        query_logs = pd.DataFrame(
            {"sample_id": datamodule.unlabeled_pool_indices[:samples_evaluated],
             "metric": metric.tolist()},
        )

        query_logs = query_logs.sort_values("metric", ascending=False)
        return query_logs

    def evaluate_metric(self, model, datamodule, num_samples_to_evaluate: int):
        """
        Estimate the uncertainty of the first <num_samples_to_evaluate> elements of the unlabeled pool
        :param num_samples_to_evaluate: The number of samples to evaluate from the unlabeled pool
        :return: Numpy array of uncertainties of the evaluated samples
        """
        self.logger.info("Evaluating metric!")

        sampler_indices = datamodule.unlabeled_pool_indices[:num_samples_to_evaluate]

        sampler = self._get_sampler(sampler_indices, model.feature_model, datamodule.unaugmented_dataset_train)

        sampler_indices_order = {sample_idx: idx for idx, sample_idx in
                                 zip(range(len(sampler_indices)), sampler_indices)}

        unlabeled_loader = DataLoader(IndexedDataset(datamodule.unaugmented_dataset_train),
                                      batch_size=datamodule.batch_size,
                                      sampler=sampler,
                                      num_workers=datamodule.num_workers,
                                      drop_last=False,
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True,
                                      collate_fn=id_collate)

        model.eval()
        metrics = torch.tensor([], device=model.device)
        evaluated_sample_indices = []

        self.logger.debug(
            f"Size of unlabelled loader: {len(unlabeled_loader)}")

        # tqdm(unlabeled_loader, leave=False, total=len(unlabeled_loader), mininterval=10, desc="Evaluating")
        self.logger.info("Evaluating...")
        with torch.no_grad():
            for ids, (inputs, _) in unlabeled_loader:
                inputs = inputs.to(model.device)

                metric = model(inputs)
                metrics = torch.cat((metrics, metric), 0)

                evaluated_sample_indices.extend(ids)

        model.train()

        metrics = np.squeeze(metrics.cpu().numpy())
        evaluated_sample_indices = [sampler_indices_order[idx] for idx in evaluated_sample_indices]
        sorted_key_idxs = np.argsort(evaluated_sample_indices).tolist()
        metrics = metrics[sorted_key_idxs]

        return metrics

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


class ModelQueryHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: ModelQueryHyperParameterSet = ModelQueryHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> ModelQueryHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)
        return hyperparam_set


class ModelQueryDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: ModelQueryHyperParameterSpace = ModelQueryHyperParameterSpace()):
        super().__init__(QueryType.ModelQuery, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> ModelQueryDefinition:
        return ModelQueryDefinition(self.hyperparam_space.suggest(trial))
