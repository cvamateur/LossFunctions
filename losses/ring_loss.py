import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA


class RingLoss(nn.Module):
    """
    Refer to paper [Ring loss: Convex Feature Normalization for Face Recognition]
    (https://arxiv.org/pdf/1803.00130.pdf).

    This loss accepts a 2D Tensor of shape [N, D] as features. Returns ring-loss
    which plays a role as an auxiliary loss that can be used together with SoftmaxLoss,
    CenterLoss, TripletLoss, etc.

    @Usage:
    --------
    >> criterion1 = SoftmaxLoss()
    >> criterion2 = RingLoss(0.01)
    >> logits, feats = (model)
    >> loss = criterion1(logits) + criterion2(feats)
    >> loss.backward()
    """

    def __init__(self, r: float = 1.0, reduction="mean"):
        assert reduction in ["mean", "sum", "none"], f"reduction must be 'mean', 'sum' or 'none'"
        super(RingLoss, self).__init__()
        self._reduction = reduction
        self.R = nn.Parameter(torch.tensor([r], dtype=torch.float32))

    def forward(self, feats):
        feats_norm = LA.norm(feats, dim=1).clip(min=1e-8)
        losses = torch.square(feats_norm - self.R).mul(0.5)
        if self._reduction == "mean":
            return torch.mean(losses)
        elif self._reduction == "sum":
            return torch.sum(losses)
        else:
            return losses

