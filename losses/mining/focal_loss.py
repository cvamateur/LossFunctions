import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List


class FocalLoss(nn.Module):
    """
    Refer to paper [Focal Loss for Dense Object Detection]
    (https://arxiv.org/pdf/1708.02002v2.pdf)

    For official sigmoid_focal_loss() function, refer to [LINK]
    (https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html)

    This is a multi-class focal loss that extents the official binary version.

    @Parameters:
    ------------
    alpha (List[float] | Tensor[float]):
        Weighting factor in range (0, 1) for each class.
    gamma (float):
        Exponent of the modulating factor (1-pt) to balance easy vs. hard examples.
        Default: ``2``.
    reduction (str):
        ``none`` | ``mean`` | ``sum``
        ``none``: No reduction will be applied to the output.
        ``mean``: The output will be averaged.
        ``sum``: The output will be summed.
        Default: ``mean``.

    logits:
        Shape of [N, C] or [N, C, D0, D1, ... Dk]
    labels:
        Shape of [N] or [N, D0, D1, ... Dk]
    """

    def __init__(self, alpha: Union[List[float], torch.Tensor] = None, gamma: float = 0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        if logits.ndim > 2:
            C = logits.size(1)
            logits = logits.permute(0, *range(2, logits.ndim), 1)
            logits = logits.reshape(-1, C)
            targets = targets.view(-1)

        if targets.size(0) == 0:
            return targets.new_zeros([1])

        N = logits.size(0)
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs[range(N), targets]
        pt = torch.exp(log_pt)
        modulating = torch.pow(1.0 - pt, self.gamma)      # (1 - pt)^gamma

        # -alpha * (1 - pt)^gamma * log(pt)
        loss = modulating * F.nll_loss(log_probs, targets, weight=self.alpha, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss
