import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA


class L_SoftmaxLinear(nn.Module):
    """
    Refer to paper [Large-Margin Softmax Loss for Convolutional Neural Networks]
    (https://arxiv.org/pdf/1612.02295.pdf).
    """

    def __init__(self, in_features: int, out_features: int, margin: int = 3, dtype=None, device=None):
        super().__init__()
        if dtype is None: dtype = torch.float32
        if device is None: device = torch.device("cpu")
        self.weight = nn.Parameter(torch.randn([out_features, in_features], dtype=dtype, device=device))
        self.bias = None
        torch.nn.init.xavier_normal_(self.weight)

        # binomial coefficients: +C_m^0, -C_m^2, +C_m^4, ...
        Cm_2n = torch.tensor([math.comb(margin, k) for k in range(0, margin + 1, 2)], dtype=dtype, device=device)
        Cm_2n[1::2].mul_(-1.0)
        pow_cos = torch.tensor([margin - k for k in range(0, margin + 1, 2)], dtype=dtype, device=device)
        pow_sin2 = torch.tensor([k for k in range(1 + margin // 2)])
        self.register_buffer("margin", torch.tensor([margin], dtype=dtype, device=device))
        self.register_buffer("Cm_2n", Cm_2n)
        self.register_buffer("pow_cos", pow_cos)
        self.register_buffer("pow_sin2", pow_sin2)
        self._beta = 100

    def _fc_impl(self, feats):
        logits = F.linear(feats, self.weight)
        return logits

    def forward(self, feats, targets=None):
        if self.training and targets is None:
            raise RuntimeError("targets is None while module in training phase")

        logits = self._fc_impl(feats)  # [N, C]
        if targets is None: return logits  # Testing Phase

        # Training Phase
        N = feats.size(0)
        indices = feats.new_tensor(range(N), dtype=torch.int64)
        logits_of_target = logits[indices, targets]

        w_norm = LA.norm(self.weight, dim=1)[targets]
        f_norm = LA.norm(feats, dim=1)
        wf = w_norm * f_norm
        cos_theta = logits_of_target / wf
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
        cos_m_theta = self.calculate_cos_m_theta(cos_theta)
        k = self.find_k(cos_theta)

        # Equation 8 in paper
        logits_of_target_new = wf * (torch.pow(-1.0, k) * cos_m_theta - 2.0 * k)
        logits[indices, targets] = (logits_of_target_new + self._beta * logits_of_target) / (1 + self._beta)
        self._beta *= 0.99
        return logits

    def calculate_cos_m_theta(self, cos_theta):
        """
        Equation 7 in paper.
        """
        cos_theta = cos_theta.view(-1, 1)  # [N, 1]
        sin2_theta = 1.0 - torch.square(cos_theta)  # [N, 1]
        cos_power_m_2n = torch.pow(cos_theta, self.pow_cos)
        sin2_power_n = torch.pow(sin2_theta, self.pow_sin2)
        cos_m_theta = torch.sum(self.Cm_2n * cos_power_m_2n * sin2_power_n, dim=1)
        return cos_m_theta

    def find_k(self, cos_theta):
        theta = torch.acos(cos_theta)
        k = torch.floor(theta * self.margin / math.pi).detach()
        return k


class A_SoftmaxLinear(L_SoftmaxLinear):
    """
    Refer to paper [SphereFace: Deep Hypersphere Embedding for Face Recognition]
    (https://arxiv.org/pdf/1704.08063.pdf).
    """

    def _fc_impl(self, feats):
        logits = F.linear(feats, F.normalize(self.weight))
        return logits


class CosFaceLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, s: float = 32.0, m: float = 0.1, dtype=None, device=None):
        super(CosFaceLinear, self).__init__()
        if dtype is None: dtype = torch.float32
        if device is None: device = torch.device("cpu")

        m_max = self.upper_bound_m(out_features, in_features)
        s_min = self.lower_bound_s(out_features)
        if m > m_max:
            raise RuntimeError(f"Angular margin is greater than its upper bound: {m_max}")
        if s < s_min:
            raise RuntimeError(f"Feature norm is less than its lower bound: {s_min}")

        self.weight = nn.Parameter(torch.randn([out_features, in_features], dtype=dtype, device=device))
        self.bias = None
        self.register_buffer("s", torch.tensor([s], dtype=dtype, device=device))
        self.register_buffer("m", torch.tensor([m], dtype=dtype, device=device))
        nn.init.xavier_normal_(self.weight)

    def forward(self, feats, targets=None):
        if self.training and targets is None:
            raise RuntimeError("targets is None while module in training phase")

        if targets is None:
            logits = F.linear(F.normalize(feats), F.normalize(self.weight))
            return logits

        N = feats.size(0)
        indices = feats.new_tensor(range(N), dtype=torch.int64)
        cos_theta = F.linear(F.normalize(feats), F.normalize(self.weight))
        cos_theta_target = cos_theta[indices, targets]
        mask_margin = torch.gt(cos_theta_target, self.m).to(feats.dtype)
        cos_theta_target_new = cos_theta_target - mask_margin * self.m
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
