import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ArcFaceLinear(nn.Module):
    """
    Refer to paper [ArcFace: Additive Angular Margin Loss for Deep Face Recognition]
    (https://arxiv.org/pdf/1801.07698v1.pdf)

    @Parameters:
    -----------
    s (float):
        Scale factor for the norm of features after normalization.

    m (float):
        Additive angular margin.

    @Usage:
    -------
    This is the last FC layer in network, work together with original SoftmaxLoss.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 32.0, m: float = 0.5):
        super(ArcFaceLinear, self).__init__()
        s_min = self._lower_s(out_features)
        if s <= s_min: raise RuntimeError(f"Norm factor s is less than lower bound: {s_min: :.2f}")
        self.register_buffer("s", torch.tensor([s], dtype=torch.float32))
        self.register_buffer("m", torch.tensor([m], dtype=torch.float32))
        self.register_buffer("pi_m", torch.tensor([math.pi - m], dtype=torch.float32))
        self.bias = None
        self.weight = nn.Parameter(torch.empty([out_features, in_features], dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feats, targets):
        N = feats.size(0)
        cos_theta = F.linear(F.normalize(feats), F.normalize(self.weight))
        indices = feats.new_tensor(range(N), dtype=torch.int64)
        with torch.no_grad():
            target_cos_theta = cos_theta[indices, targets].clamp(min=-1.0, max=1.0)
            target_theta = torch.acos(target_cos_theta)
            mask = torch.lt(target_theta, self.pi_m).to(torch.float32)
            target_theta = target_theta + mask * self.m
            cos_theta[indices, targets] = torch.cos(target_theta)
        logits = self.s * cos_theta
        return logits

    @staticmethod
    def _lower_s(C: int, Pw: float = 0.9):
        return (C - 1) / C * math.log((C - 1) * Pw / (1 - Pw))
