from abc import ABC
from typing import Callable, Optional, Any

import pandas as pd

from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet
from lowdataregime.active_learning.query.queries import Query, QueryDefinition, QueryType


class RandomQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 log_query: bool = False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.log_query = log_query


class RandomQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: RandomQueryHyperParameterSet = RandomQueryHyperParameterSet()):
        super().__init__(QueryType.RandomQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        return RandomQuery


class RandomQuery(Query):

    def sort_unlabeled_pool_by_metric(self, model, datamodule, num_samples_to_evaluate: int, ascending: bool = False):
        datamodule.sort_unlabeled_indices_randomly()
        query_logs = self.get_query_logs(datamodule, datamodule.unlabeled_pool_indices)
        return query_logs