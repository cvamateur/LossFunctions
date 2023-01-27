import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA


class L_SoftmaxLinear(nn.Module):
    """
    Refer to paper [Large-Margin Softmax Loss for Convolutional Neural Networks]
    (https://arxiv.org/pdf/1612.02295.pdf).

    @Parameters:
    ------------
    margin (int):
        A integer multiplicative angular margin.

    @Usage:
    ------
    This is the last FC layer in network, work together with original SoftmaxLoss.
    """

    def __init__(self, in_features: int, out_features: int, margin: int = 3):
        super().__init__()
        dtype, device = torch.float32, torch.device("cpu")
        self.bias = None
        self.weight = nn.Parameter(torch.randn([out_features, in_features], dtype=dtype, device=device))
        nn.init.xavier_normal_(self.weight)

        # binomial coefficients: +C_m^0, -C_m^2, +C_m^4, ...
        Cm_2n = torch.tensor([math.comb(margin, k) for k in range(0, margin + 1, 2)], dtype=dtype, device=device)
        Cm_2n[1::2].mul_(-1.0)
        pow_cos = torch.tensor([margin - k for k in range(0, margin + 1, 2)], dtype=dtype, device=device)
        pow_sin2 = torch.tensor([k for k in range(1 + margin // 2)], dtype=dtype, device=device)
        self.register_buffer("margin", torch.tensor([margin], dtype=dtype, device=device))
        self.register_buffer("Cm_2n", Cm_2n)
        self.register_buffer("pow_cos", pow_cos)
        self.register_buffer("pow_sin2", pow_sin2)

        # Coefficient of annealing algorithm
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



class SphereFaceLinear(L_SoftmaxLinear):
    """
    Refer to paper [SphereFace: Deep Hypersphere Embedding for Face Recognition]
    (https://arxiv.org/pdf/1704.08063.pdf).

    This loss is almost the same as L-Softmax Loss except the logits are calculated by
    normalized weights.
    """

    def _fc_impl(self, feats):
        logits = F.linear(feats, F.normalize(self.weight))
        return logits


# Aliases
A_SoftmaxLinear = SphereFaceLinear