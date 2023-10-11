import torch
from torch import Tensor


def _zeta(alpha: Tensor, beta: Tensor) -> Tensor:
    # pg 65
    return torch.where(alpha == 1.0, 0.0, -beta * torch.tan(torch.pi * alpha / 2))
