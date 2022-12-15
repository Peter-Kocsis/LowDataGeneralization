import copy
import os.path

import optuna
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional, Dict, Sequence

from lowdataregime.classification.log.logger import init_logger
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace, DefinitionSpace
from lowdataregime.active_learning.query.queries import Query, QueryDefinition, QueryType
from lowdataregime.utils.utils import SerializableEnum


class QueryFactorReductionMode(SerializableEnum):
    MULTIPLICATION = "multiplication"


class MultiFactorQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 query_definitions: Dict[str, QueryDefinition] = {},
                 reduction_mode: QueryFactorReductionMode = QueryFactorReductionMode.MULTIPLICATION,
                 log_query: bool = False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.query_definitions = query_definitions
        self.reduction_mode = reduction_mode
        self.log_query = log_query

    def definition_space(self):
        return MultiFactorQueryHyperParameterSpace(self)


class MultiFactorQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: MultiFactorQueryHyperParameterSet = MultiFactorQueryHyperParameterSet()):
        super().__init__(QueryType.MultiFactorQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        raise NotImplementedError()

    def instantiate(self, log_folder, *args, **kwargs):
        """Instantiates the module"""
        return MultiFactorQuery(log_folder, self.hyperparams)

    def definition_space(self):
        return MultiFactorQueryDefinitionSpace(self.hyperparams.definition_space())


class MultiFactorQuery(Query):
    def __init__(self, log_folder: str, params: MultiFactorQueryHyperParameterSet = MultiFactorQueryHyperParameterSet()):
        super().__init__(params)
        self.log_folder = log_folder
        self.logger = init_logger(self.__class__.__name__)

        self.queries = {key: query_def.instantiate(self.log_folder) for key, query_def in self.params.query_definitions.items()}

    @property
    def query_file(self):
        return os.path.join(self.log_folder, "multifactor_query.csv")

    def query(self, model, datamodule, num_samples_to_query, num_samples_to_evaluate, ascending: bool = False):
        datamodule.sort_unlabeled_indices_randomly()
        query_logs = self.sort_unlabeled_pool_by_metric(model, datamodule, num_samples_to_evaluate, ascending)
        query_logs["chosen"] = pd.Series(
            [True] * num_samples_to_query + [False] * (len(query_logs) - num_samples_to_query))
        if self.params.log_query:
            self.log_query(query_logs)
        datamodule.label_samples(num_samples_to_query, model.feature_model, model.device)

    def log_query(self, query_logs: pd.DataFrame):
        query_logs.to_csv(self.query_file, index=False)

    def sort_unlabeled_pool_by_metric(self, model, datamodule, num_samples_to_evaluate: int, ascending: bool = False):
        """
        Sorts the first <num_samples_to_evaluate> samples of the unlabeled pool by attention
        :param num_samples_to_evaluate: The number of unlabeled samples to evaluate
        :return: Logs if requested
        """
        self.logger.info(f"Sorting the unlabeled pool by {list(self.queries.keys())}!")
        metric, metrics = self.evaluate_metric(model, datamodule, num_samples_to_evaluate)
        if ascending:
            metric = -metric
            for key in metrics:
                metrics[key] = -metrics[key]
        query_logs = self.get_query_logs(datamodule, (metric, metrics))

        datamodule.sort_unlabeled_indices_by_list(metric)

        self.logger.info(f"Unlabeled pool sorted by {list(self.queries.keys())}!")
        return query_logs

    def get_query_logs(self, datamodule, all_metrics):
        final_metric, metrics = all_metrics
        query_logs = {}

        for key, metric in metrics.items():
            samples_evaluated = len(metric)
            query_logs["sample_id"] = datamodule.unlabeled_pool_indices[:samples_evaluated]
            query_logs[key] = metric.tolist()

        query_logs["final_metric"] = final_metric

        query_logs = pd.DataFrame(query_logs)
        query_logs = query_logs.sort_values("final_metric", ascending=False)
        return query_logs

    def evaluate_metric(self, model, datamodule, num_samples_to_evaluate: int):
        metrics = {key: query.evaluate_metric(model, datamodule, num_samples_to_evaluate) for key, query in self.queries.items()}
        metric = self.reduce_metrics(metrics)
        return metric, metrics

    def reduce_metrics(self, metrics):
        if len(metrics) == 0:
            return None
        if self.params.reduction_mode == QueryFactorReductionMode.MULTIPLICATION:
            return np.prod(np.array(list(metrics.values())), axis=0)
        raise NotImplementedError()


class MultiFactorQueryHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: MultiFactorQueryHyperParameterSet = MultiFactorQueryHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> MultiFactorQueryHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)
        return hyperparam_set


class MultiFactorQueryDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: MultiFactorQueryHyperParameterSpace = MultiFactorQueryHyperParameterSpace()):
        super().__init__(QueryType.MultiFactorQuery, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> MultiFactorQueryDefinition:
        return MultiFactorQueryDefinition(self.hyperparam_space.suggest(trial))
