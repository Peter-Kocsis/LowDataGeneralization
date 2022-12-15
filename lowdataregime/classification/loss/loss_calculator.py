from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Sequence

import optuna
import torch
import torch.nn as nn

from lowdataregime.classification.loss.losses import LossDefinition
from lowdataregime.parameters.params import HyperParameterSet, DefinitionSet, HyperParameterSpace


class LossEvaluatorHyperParameterSet(HyperParameterSet):
    def __init__(self,
                 layers_needed: Dict[str, str] = None,
                 target_needed: bool = True,
                 loss_definition: LossDefinition = None,
                 weight: float = 1,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.layers_needed = layers_needed
        self.target_needed = target_needed
        self.loss_definition = loss_definition
        self.weight = weight


class LossEvaluator(nn.Module):
    def __init__(self, params: LossEvaluatorHyperParameterSet):
        super(LossEvaluator, self).__init__()
        self.params = params
        self.loss_layer = self.params.loss_definition.instantiate()

        self._inspected_layers = {}
        self._inspection_handlers = {}

        self._requires_layers = self.params.layers_needed is not None

    def inspect_layers(self, model):
        if not self._requires_layers:
            return
        self._inspected_layers = {}
        self._inspection_handlers = {key: model.inspect_layer_output(layer_to_inspect,
                                                                     name=key,
                                                                     storage_dict=self._inspected_layers,
                                                                     unsqueeze=False)
                                     for key, layer_to_inspect in self.params.layers_needed.items()}

    def remove_inspection(self):
        if not self._requires_layers:
            return
        self._inspection_handlers = {key: handler.remove() if handler is not None else None
                                     for key, handler in self._inspection_handlers.items()}
        self._inspected_layers = {}

    def forward(self, logits, y):
        if self._requires_layers:
            if self.params.target_needed:
                loss = self.loss_layer(**self._inspected_layers, target=y)
            else:
                loss = self.loss_layer(**self._inspected_layers)
        else:
            loss = self.loss_layer(logits, y)
        return loss * self.params.weight


class LossCalculator(nn.Module):
    def __init__(self, loss_calc_params: Dict[str, LossEvaluatorHyperParameterSet]):
        super(LossCalculator, self).__init__()
        self.loss_calc_params = loss_calc_params
        self.loss_evaluators = nn.ModuleDict(
            OrderedDict([(key, LossEvaluator(params)) for key, params in self.loss_calc_params.items()])) if self.loss_calc_params is not None else None

    def inspect_layers(self, model):
        for loss_evaluator in self.loss_evaluators.values():
            loss_evaluator.inspect_layers(model)

    def remove_inspection(self):
        for loss_evaluator in self.loss_evaluators.values():
            loss_evaluator.remove_inspection()

    def forward(self, logits, y):
        assert len(self.loss_evaluators) > 0, "No loss evaluator defined, unable to calculate the loss!"
        losses = {key: loss_evaluator(logits, y) for key, loss_evaluator in self.loss_evaluators.items()}
        return torch.sum(torch.stack(list(losses.values())), dim=0), losses
