import torch
import torch.nn as nn
import torch.nn.functional as F



class NormFaceLinear(nn.Module):
    """
    Refer to paper [NormFace: L2 Hypersphere Embedding for Face Verification]
    (https://arxiv.org/pdf/1704.06369.pdf)

    @Parameters:
    -----------
    s (float):
        Scale factor for the norm of features after normalization.

    @Usage:
    -------
    Replace the last FC layer with NormFaceLinear, the original SoftmaxLoss will
    become `Normalizes Softmax Loss (NSL)`.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 32.0):
        super(NormFaceLinear, self).__init__()
        self.register_buffer("s", torch.tensor([s], dtype=torch.float32))
        self.bias = None
        self.weight = nn.Parameter(torch.empty([out_features, in_features], dtype=torch.float32))
        nn.init.normal_(self.weight, 0.0, 0.01)

    def forward(self, feats):
        cos_theta = F.linear(F.normalize(feats), F.normalize(self.weight))
        logits = self.s * cos_theta
        return logits
