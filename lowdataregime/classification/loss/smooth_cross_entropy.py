"""
Source: https://github.com/davda54/sam/blob/main/example/model/smooth_cross_entropy.py
"""
import torch
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1):
    """
    Smooth Cross Entropy
    Adapted from: https://github.com/davda54/sam/blob/main/example/model/smooth_cross_entropy.py
    """
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)