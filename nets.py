import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA


class MNIST_Net(nn.Module):
    """
    2D features extractor for MNIST.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super(MNIST_Net, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 16, 3, 1, 1)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv0(x), inplace=True)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 3, 2, 1)

        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.max_pool2d(x, 3, 2, 1)

        x = F.relu(self.conv4(x), inplace=True)
        x = self.conv5(x)
        feats = self.pool(x)
        return feats.view(-1, 2)


class NormLinear(nn.Module):
    """
    A Linear layer that normalize weights.
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool = False, dtype=None, device=None):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros([out_features, in_features], dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros([out_features], dtype=dtype, device=device)) if use_bias else None
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        out = F.linear(x, F.normalize(self.weight), self.bias)
        return out


class L2NormLayer(nn.Module):
    """
    Normalize and Rescale input.

    NOTE:
        For L2-Softmax Loss, alpha should be at least log(p *(C-2)/(1-p)).
    """

    def __init__(self, alpha: float = 1.0, train_alpha: bool = False, dtype=torch.float32):
        super(L2NormLayer, self).__init__()
        if train_alpha:
            self.register_parameter("alpha", nn.Parameter(torch.tensor(alpha, dtype=dtype)))
        else:
            alpha_low = self.lower_bound(0.9, 10)
            assert alpha > alpha_low, f"Alpha must be at least: {alpha_low:.1f}"
            self.register_buffer("alpha", torch.tensor(alpha, dtype=dtype))

    def forward(self, x):
        x = self.alpha * F.normalize(x)
        return x

    @staticmethod
    def lower_bound(p, c) -> float:
        return math.log(p * (c - 2) / (1 - p))


class L_SoftmaxLinear(nn.Module):
    """
    Refer to paper [Large-Margin Softmax Loss for Convolutional Neural Networks]
    (https://arxiv.org/pdf/1612.02295.pdf).

    Refer to paper [SphereFace: Deep Hypersphere Embedding for Face Recognition]
    (https://arxiv.org/pdf/1704.08063.pdf).
    """

    def __init__(self, in_features: int, out_features: int, margin: int = 1, dtype=None, device=None):
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

    def forward(self, feats, targets=None):
        if self.training and targets is None:
            raise RuntimeError("targets is None while module in training phase")

        logits = F.linear(feats, self.weight)   # [N, C]
        if targets is None: return logits       # Testing Phase

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
        cos_theta = cos_theta.view(-1, 1)           # [N, 1]
        sin2_theta = 1.0 - torch.square(cos_theta)   # [N, 1]
        cos_power_m_2n = torch.pow(cos_theta, self.pow_cos)
        sin2_power_n = torch.pow(sin2_theta, self.pow_sin2)
        cos_m_theta = torch.sum(self.Cm_2n * cos_power_m_2n * sin2_power_n, dim=1)
        return cos_m_theta

    def find_k(self, cos_theta):
        theta = torch.acos(cos_theta)
        k = torch.floor(theta * self.margin / math.pi).detach()
        return k
