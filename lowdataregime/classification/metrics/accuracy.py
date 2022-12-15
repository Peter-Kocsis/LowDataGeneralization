from typing import Optional, Callable, Any

import torch
from pytorch_lightning.metrics import Accuracy


class TopKAccuracy(Accuracy):
    def __init__(self,
                 top_k: int = 1,
                 threshold: float = 0.5,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None):
        super().__init__(threshold, compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.top_k = top_k

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape[0] == target.shape[0]

        _, preds = preds.topk(self.top_k, 1, True, True)

        correct = preds.eq(target.unsqueeze(1))

        correct_k = correct.sum()

        self.correct += correct_k
        self.total += target.numel()
