import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):
    """
    Refer to Pytorch built-in loss function: nn.CrossEntropyLoss().

    This is the Original Softmax Loss (Cross-Entropy Loss).

    This loss accepts a 2D Tensor of shape [N, C] as logits, which
    are scores output from a FC layer without softmax, and a 1D
    Tensor of shape [N] as targets.

    @Usage:
    -------
    >> criterion = SoftmaxLoss()
    >> logits = model(inputs)
    >> loss = criterion(logits, targets)
    >> loss.backward()
    """

    def __init__(self, weight=None, reduction="mean"):
        super(SoftmaxLoss, self).__init__()
        self._reduction = reduction
        self._weight = weight

    def forward(self, logits, targets):
        out = F.log_softmax(logits, dim=1)
        out = F.nll_loss(out, targets, weight=self._weight, reduction=self._reduction)
        return out
