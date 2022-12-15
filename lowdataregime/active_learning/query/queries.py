import os
from abc import ABC, abstractmethod

import pandas as pd

from lowdataregime.parameters.params import DefinitionSet, HyperParameterSet
from lowdataregime.utils.utils import SerializableEnum


class QueryType(SerializableEnum):
    """Definition of the available query strategies"""
    RandomQuery = "RandomQuery"
    UncertaintyQuery = "UncertaintyQuery"
    CoreSetQuery = "CoreSetQuery"
    DivRankQuery = "DivRankQuery"
    AttentionQuery = "AttentionQuery"
    MultiFactorQuery = "MultiFactorQuery"
    MultiStageQuery = "MultiStageQuery"

    ModelQuery = "ModelQuery"


class QueryHyperParameterSet(HyperParameterSet):
    pass


class QueryDefinition(DefinitionSet, ABC):

    def __init__(self, type: QueryType = None, hyperparams: HyperParameterSet = None):
        super().__init__(type, hyperparams)

    def instantiate(self, *args, **kwargs):
        """Instantiates the module"""
        return self._instantiate_func(self.hyperparams)


class Query(ABC):
    def __init__(self, params: QueryHyperParameterSet = QueryHyperParameterSet()):
        self.params = params

    @property
    def query_file(self):
        raise NotImplementedError()

    def query(self, model, datamodule, num_samples_to_query, num_samples_to_evaluate, ascending: bool = False):
        datamodule.sort_unlabeled_indices_randomly()
        query_logs = self.sort_unlabeled_pool_by_metric(model, datamodule, num_samples_to_evaluate, ascending)
        query_logs["chosen"] = pd.Series(
            [True] * num_samples_to_query + [False] * (len(query_logs) - num_samples_to_query))
        if self.params.log_query:
            self.log_query(query_logs)
        datamodule.label_samples(num_samples_to_query, model.feature_model, model.device)

    def evaluate_metric(self, model, datamodule, num_samples_to_evaluate):
        raise NotImplementedError()

    def get_query_logs(self, datamodule, evaluated_metrics):
        samples_evaluated = len(evaluated_metrics)
        query_logs = pd.DataFrame(
            {"sample_id": datamodule.unlabeled_pool_indices[:samples_evaluated]},
        )

        return query_logs

    def log_query(self, query_logs: pd.DataFrame):
        query_logs.to_csv(self.query_file, index=False)

    def sort_unlabeled_pool_by_metric(self, model, datamodule, num_samples_to_evaluate: int, ascending):
        raise NotImplementedError()