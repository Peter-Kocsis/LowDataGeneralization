from typing import Any

from torch import nn
from torch.autograd import Function


class _GradientGateFunction(Function):

    @staticmethod
    def forward(ctx, x, gradient_multiplier) -> Any:
        ctx.gradient_multiplier = gradient_multiplier
        return x

    @staticmethod
    def backward(ctx, grad_output) -> Any:
        return ctx.gradient_multiplier * grad_output, None


class GradientGate(nn.Module):

    def __init__(self, gradient_multiplier: float = 0.0):
        super(GradientGate, self).__init__()
        self.gradient_multiplier = gradient_multiplier

    def forward(self, x):
        x = _GradientGateFunction.apply(x, self.gradient_multiplier)
        return x

    def extra_repr(self):
        return f"gradient_multiplier: {self.gradient_multiplier}"
