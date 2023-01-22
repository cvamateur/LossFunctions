import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLinear(nn.Module):
    """
    Refer to paper [CosFace: Large Margin Cosine Loss for Deep Face Recognition]
    (https://arxiv.org/pdf/1801.09414.pdf)

    @Parameters:
    -----------
    s (float):
        Scale factor for the norm of features after normalization.

    m (float):
        Additive cosine angular margin.

    @Usage:
    -------
    This is the last FC layer in network, work together with original SoftmaxLoss.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 32.0, m: float = 0.35):
        super(CosFaceLinear, self).__init__()
        dtype, device = torch.float32, torch.device("cpu")
        m_max = self.upper_bound_m(out_features, in_features)
        s_min = self.lower_bound_s(out_features)
        if m > m_max:
            raise RuntimeError(f"Angular margin is greater than its upper bound: {m_max}")
        if s < s_min:
            raise RuntimeError(f"Feature norm is less than its lower bound: {s_min}")
        self.bias = None
        self.weight = nn.Parameter(torch.randn([out_features, in_features], dtype=dtype, device=device))
        self.register_buffer("s", torch.tensor([s], dtype=dtype, device=device))
        self.register_buffer("m", torch.tensor([m], dtype=dtype, device=device))
        nn.init.xavier_normal_(self.weight)

    def forward(self, feats, targets):
        N = feats.size(0)
        indices = feats.new_tensor(range(N), dtype=torch.int64)
        cos_theta = F.linear(F.normalize(feats), F.normalize(self.weight))
        cos_theta_target = cos_theta[indices, targets]
        cos_theta_target_new = cos_theta_target - self.m
        cos_theta[indices, targets] = cos_theta_target_new
        logits = cos_theta * self.s
        return logits

    @staticmethod
    def lower_bound_s(C: int, Pw: float = 0.9):
        return (C - 1) / C * math.log((C - 1) * Pw / (1 - Pw))

    @staticmethod
    def upper_bound_m(C: int, K: int):
        if K == 2:
            return 1 - math.cos(2 * math.pi / C)
        else:
            return C / (C - 1)


AM_SoftmaxLinear = CosFaceLinear
