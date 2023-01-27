import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Refer to paper [Supervised Contrastive Learning]
    (https://arxiv.org/pdf/2004.11362.pdf)

    Refer to paper [A Simple Framework for Contrastive Learning of Visual Representations]
    (https://arxiv.org/pdf/2002.05709.pdf)
    """
    def __init__(self, temperature: float):
        super(SupConLoss, self).__init__()

