from typing import Any, Callable, Optional
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from lowdataregime.classification.sampling.sampler import SubsetSequentialSampler
from lowdataregime.classification.log.logger import init_logger
from lowdataregime.active_learning.query.queries import Query, QueryDefinition, QueryType
import numpy as np
import networkx as nx
import scipy
from networkx.exception import NetworkXError

from lowdataregime.parameters.params import HyperParameterSet


class DivRankQueryHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 num_of_neighbors: int = 3,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.num_of_neighbors = num_of_neighbors


class DivRankQueryDefinition(QueryDefinition):

    def __init__(self, hyperparams: DivRankQueryHyperParameterSet = DivRankQueryHyperParameterSet()):
        super().__init__(QueryType.DivRankQuery, hyperparams)

    @property
    def _instantiate_func(self) -> Optional[Callable]:
        raise NotImplementedError()

    def instantiate(self, *args, **kwargs):
        """Instantiates the module"""
        return DivRankQuery(self.hyperparams)


class DivRankQuery(Query):
    def __init__(self, params: DivRankQueryHyperParameterSet = DivRankQueryHyperParameterSet()):
        super().__init__(params)
        self.logger = init_logger(self.__class__.__name__)

    def query(self, model, datamodule, num_samples_to_query, num_samples_to_evaluate, ascending: bool = False):
        datamodule.sort_unlabeled_indices_randomly()

        self._sort_unlabeled_indices_by_uncertainty(model, datamodule, num_samples_to_evaluate)
        datamodule.label_samples(num_samples_to_query, model.feature_model, model.device)

    def _sort_unlabeled_indices_by_uncertainty(self, model, datamodule, num_samples_to_evaluate: int):
        """
        Sorts the first <num_samples_to_evaluate> samples of the unlabeled pool by uncertainty
        :param num_samples_to_evaluate: The number of unlabeled samples to evaluate
        :return: Logits or uncertainties if requested
        """
        self.logger.info("Sorting the unlabeled pool by uncertainty!")
        scores = self._get_divrank_scores(model, datamodule, num_samples_to_evaluate)
        datamodule.sort_unlabeled_indices_by_list(scores)
        self.logger.info("Unlabeled pool sorted by uncertainty!")

    def _get_divrank_scores(self, model, datamodule, num_samples_to_evaluate: int):
        """
        Estimate the uncertainty of the first <num_samples_to_evaluate> elements of the unlabeled pool
        :param num_samples_to_evaluate: The number of samples to evaluate from the unlabeled pool
        :return: Numpy array of uncertainties of the evaluated samples
        """

        unlabeled_loader = DataLoader(datamodule.unaugmented_dataset_train,
                                      batch_size=datamodule.batch_size,
                                      sampler=SubsetSequentialSampler(
                                          datamodule.unlabeled_pool_indices[:num_samples_to_evaluate]),
                                      num_workers=datamodule.num_workers,
                                      drop_last=True,
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True)

        model.eval()
        uncertainties = torch.tensor([], device=model.device)
        latent_positions = torch.tensor([], device=model.device)

        self.logger.debug(
            f"Size of unlabelled loader: {len(unlabeled_loader)}")

        # tqdm(unlabeled_loader, leave=False, total=len(unlabeled_loader), mininterval=10, desc="Evaluating")
        self.logger.info("Evaluating...")
        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                features = model.features(inputs)
                scores = model.head(features)
                probs = F.softmax(scores, dim=1)
                log_probs = torch.log(probs)
                to_be_summed = probs * log_probs
                entropy = -torch.sum(to_be_summed, dim=1, keepdim=True)

                uncertainties = torch.cat((uncertainties, entropy), 0)
                latent_positions = torch.cat((latent_positions, features), 0)

        model.train()

        graph = nx.Graph()
        features = features.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()
        dist = scipy.spatial.distance.cdist(features, features)
        k = self.params.num_of_neighbors
        for n1 in range(len(features)):
            nearest_points = np.argpartition(dist[n1], k)[:k]
            for n2 in nearest_points:
                if n1 != n2:
                    graph.add_edge(n1, n2, weight=1/(dist[n1, n2] + 1e-7))

        node_weights = {idx: float(entropy) for idx, entropy in enumerate(uncertainties)}
        scores = divrank(graph, personalization=node_weights)

        return [scores[n] for n in range(len(scores))]


def divrank(G, alpha=0.25, d=0.85, personalization=None,
            max_iter=1000, tol=1.0e-6, nstart=None, weight='weight',
            dangling=None):
    '''
    Returns the DivRank (Diverse Rank) of the nodes in the graph.
    This code is based on networkx.pagerank.

    Args: (diff from pagerank)
      alpha: controls strength of self-link [0.0-1.0]
      d: the damping factor

    Reference:
      Qiaozhu Mei and Jian Guo and Dragomir Radev,
      DivRank: the Interplay of Prestige and Diversity in Information Networks,
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.174.7982
    '''

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # self-link (DivRank)
    for n in W.nodes():
        for n_ in W.nodes():
            if n != n_:
                if n_ in W[n]:
                    W[n][n_][weight] *= alpha
            else:
                if n_ not in W[n]:
                    W.add_edge(n, n_)
                W[n][n_][weight] = 1.0 - alpha

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            D_t = sum(W[n][nbr][weight] * xlast[nbr] for nbr in W[n])
            for nbr in W[n]:
                # x[nbr] += d * xlast[n] * W[n][nbr][weight]
                x[nbr] += (
                        d * (W[n][nbr][weight] * xlast[nbr] / D_t) * xlast[n]
                )
            x[n] += danglesum * dangling_weights[n] + (1.0 - d) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise NetworkXError('divrank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)
