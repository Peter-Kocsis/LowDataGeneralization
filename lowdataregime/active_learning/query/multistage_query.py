import copy
import os.path

import optuna
import pandas as pd
from typing import Any, Callable, Optional, Dict, Sequence, List

from lowdataregime.classification.log.logger import init_logger
from lowdataregime.parameters.params import HyperParameterSet, HyperParameterSpace, DefinitionSpace
from lowdataregime.active_learning.query.queries import Query, QueryDefinition, QueryType
from functools import reduce


class MultiStageQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 query_definitions: Dict[str, QueryDefinition] = {},
                 num_samples_of_query_stages: List[int] = [],
                 log_query: bool = False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.query_definitions = query_definitions
        self.num_samples_of_query_stages = num_samples_of_query_stages
        self.log_query = log_query

    def definition_space(self):
        return MultiStageQueryHyperParameterSpace(self)


class MultiStageQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: MultiStageQueryHyperParameterSet = MultiStageQueryHyperParameterSet()):
        super().__init__(QueryType.MultiStageQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        raise NotImplementedError()

    def instantiate(self, log_folder, *args, **kwargs):
        """Instantiates the module"""
        return MultiStageQuery(log_folder, self.hyperparams)

    def definition_space(self):
        return MultiStageQueryDefinitionSpace(self.hyperparams.definition_space())


class MultiStageQuery(Query):
    def __init__(self, log_folder: str, params: MultiStageQueryHyperParameterSet = MultiStageQueryHyperParameterSet()):
        super().__init__(params)
        self.log_folder = log_folder
        self.logger = init_logger(self.__class__.__name__)

        self.queries = {key: query_def.instantiate(self.log_folder) for key, query_def in
                        self.params.query_definitions.items()}

    @property
    def query_file(self):
        return os.path.join(self.log_folder, "multistage_query.csv")

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
        query_stage_sizes = [num_samples_to_evaluate] + self.params.num_samples_of_query_stages
        query_logs = {}
        for query_stage_size, (query_key, query) in zip(query_stage_sizes, self.queries.items()):
            query_logs[query_key] = query.sort_unlabeled_pool_by_metric(model, datamodule, query_stage_size, ascending)

        query_logs = reduce(lambda left, right: pd.merge(left, right, how="outer", on='sample_id'), query_logs.values())
        self.logger.info(f"Unlabeled pool sorted by {list(self.queries.keys())}!")
        return query_logs


class MultiStageQueryHyperParameterSpace(HyperParameterSpace):

    def __init__(self, default_hyperparam_set: MultiStageQueryHyperParameterSet = MultiStageQueryHyperParameterSet()):
        self.default_hyperparam_set = default_hyperparam_set

    @property
    def search_grid(self) -> Dict[str, Sequence[Any]]:
        return {}

    @property
    def search_space(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError()

    def suggest(self, trial: optuna.Trial) -> MultiStageQueryHyperParameterSet:
        hyperparam_set = copy.deepcopy(self.default_hyperparam_set)
        return hyperparam_set


class MultiStageQueryDefinitionSpace(DefinitionSpace):
    """DefinitionSpace of the GeneralNet"""

    def __init__(self, hyperparam_space: MultiStageQueryHyperParameterSpace = MultiStageQueryHyperParameterSpace()):
        super().__init__(QueryType.MultiStageQuery, hyperparam_space)

    def suggest(self, trial: optuna.Trial) -> MultiStageQueryDefinition:
        return MultiStageQueryDefinition(self.hyperparam_space.suggest(trial))
