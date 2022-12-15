from typing import Any, Callable, Optional
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader

from lowdataregime.classification.sampling.sampler import SubsetSequentialSampler
from lowdataregime.classification.log.logger import init_logger
from lowdataregime.active_learning.query.queries import Query, QueryDefinition, QueryType
import numpy as np

from lowdataregime.parameters.params import HyperParameterSet

from lowdataregime.utils.distance import CosineDistanceDefinitionSet, EuclideanDistanceDefinitionSet, DistanceDefinitionSet, \
    Distance


class CoreSetQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 layer_of_features: str = None,
                 distance_metric: DistanceDefinitionSet = EuclideanDistanceDefinitionSet(),
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.layer_of_features = layer_of_features
        self.distance_metric = distance_metric


class CoreSetQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: CoreSetQueryHyperParameterSet = CoreSetQueryHyperParameterSet()):
        super().__init__(QueryType.CoreSetQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        raise NotImplementedError()

    def instantiate(self, *args, **kwargs):
        """Instantiates the module"""
        return CoreSetQuery(self.hyperparams)


class CoreSetQuery(Query):
    """
    Adapted from https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    """
    def __init__(self, params: CoreSetQueryHyperParameterSet = CoreSetQueryHyperParameterSet()):
        super().__init__(params)
        self.logger = init_logger(self.__class__.__name__)
        self.distance_metric: Distance = self.params.distance_metric.instantiate()

    def query(self, model, datamodule, num_samples_to_query, num_samples_to_evaluate, ascending: bool = False):
        datamodule.sort_unlabeled_indices_randomly()

        self._sort_unlabeled_indices_by_core_set(model, datamodule, num_samples_to_query, num_samples_to_evaluate)
        datamodule.label_samples(num_samples_to_query, model.feature_model, model.device)

    def _sort_unlabeled_indices_by_core_set(self, model, datamodule, num_samples_to_query, num_samples_to_evaluate):
        # use the learned representation for the k-greedy-center algorithm:
        labeled_pool_representation = self._get_latent_space(model, datamodule, num_samples_to_evaluate=None, labeled=True)
        unlabeled_pool_representation = self._get_latent_space(model, datamodule, num_samples_to_evaluate, labeled=False)
        selected_indices = self._greedy_k_center(labeled_pool_representation.cpu(), unlabeled_pool_representation.cpu(), num_samples_to_query)

        num_samples_to_evaluate = unlabeled_pool_representation.size()[0]

        sorting_values = np.zeros(num_samples_to_evaluate)
        sorting_values[selected_indices] = 1
        datamodule.sort_unlabeled_indices_by_list(sorting_values)

    def _greedy_k_center(self, labeled_representation, unlabeled_representation, num_samples_to_query):
        distance_metric = self.distance_metric
        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(
            distance_metric.get_distance_matrix(
                labeled_representation[0, :].reshape((1, labeled_representation.shape[1])),
                unlabeled_representation).detach().cpu().numpy(),
            axis=0)

        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled_representation.shape[0], 100):
            if j + 100 < labeled_representation.shape[0]:
                dist = distance_metric.get_distance_matrix(
                    labeled_representation[j:j + 100, :],
                    unlabeled_representation).detach().cpu().numpy()
            else:
                dist = distance_metric.get_distance_matrix(
                    labeled_representation[j:, :],
                    unlabeled_representation).detach().cpu().numpy()
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(num_samples_to_query - 1):
            dist = distance_metric.get_distance_matrix(
                unlabeled_representation[greedy_indices[-1], :].reshape((1, unlabeled_representation.shape[1])),
                unlabeled_representation).detach().cpu().numpy()
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def _get_latent_space(self, model, datamodule, num_samples_to_evaluate: int, labeled: bool):
        if labeled:
            loader = DataLoader(datamodule.unaugmented_dataset_train,
                                batch_size=datamodule.batch_size,
                                sampler=SubsetSequentialSampler(
                                    datamodule.labeled_pool_indices[:num_samples_to_evaluate]),
                                num_workers=datamodule.num_workers,
                                drop_last=False,
                                # more convenient if we maintain the order of subset
                                pin_memory=True)
        else:
            loader = DataLoader(datamodule.unaugmented_dataset_train,
                                batch_size=datamodule.batch_size,
                                sampler=SubsetSequentialSampler(
                                              datamodule.unlabeled_pool_indices[:num_samples_to_evaluate]),
                                num_workers=datamodule.num_workers,
                                drop_last=False,
                                # more convenient if we maintain the order of subset
                                pin_memory=True)

        model.eval()

        model.clean_inspected_layers()
        handle = model.inspect_layer_output(self.params.layer_of_features, unsqueeze=False)
        self.logger.debug("Feature inspection is registered!")

        self.logger.debug(f"Size of query loader: {len(loader)}")

        self.logger.info("Evaluating...")
        with torch.no_grad():
            for (inputs, labels) in loader:
                inputs = inputs.to(model.device)
                output = model(inputs)

        model.train()

        handle.remove()
        features = model.inspected_variables[self.params.layer_of_features].detach()
        model.clean_inspected_layers()

        return features
