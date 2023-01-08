import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_Net(nn.Module):

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

    def __init__(self, in_channels: int = 2, use_bias: bool = False, dtype=torch.float32):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros([in_channels, 10], dtype=dtype))
        self.bias = nn.Parameter(torch.zeros([10], dtype=dtype)) if use_bias else None
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

    def __init__(self, alpha: float = 1.0, learnable: bool = False, dtype=torch.float32):
        super(L2NormLayer, self).__init__()
        if learnable:
            self.register_parameter("alpha", nn.Parameter(torch.tensor(alpha, dtype=dtype)))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=dtype))

    def forward(self, x):
        x = self.alpha * F.normalize(x)
        return x


def lower_bound(p, c) -> float:
    return math.log(p * (c - 2) / (1 - p))
