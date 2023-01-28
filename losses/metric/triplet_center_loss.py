import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function


class TripletCenterLoss(nn.Module):
    """
    Refer to paper [Triplet-Center Loss for Multi-View 3D Object Retrieval]
    (https://arxiv.org/pdf/1803.06189.pdf)
    """

    def __init__(self, feat_dims: int, num_classes: int, margin: float = 1.0):
        super(TripletCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.zeros([num_classes, feat_dims], dtype=torch.float32))
        self.register_buffer("margin", torch.tensor([margin], dtype=torch.float32))

    def forward(self, feats, targets):
        return TripletCenterLossFunc.apply(feats, self.centers, targets, self.margin)


class TripletCenterLossFunc(Function):

    @staticmethod
    def forward(ctx, feats, centers, targets, margin):
        """
        Equation 3.
        """
        # Difference of features with all centers
        diff_mat = feats.unsqueeze(1) - centers                     # [B, C, 2]
        dist_mat = 0.5 * diff_mat.square().sum(dim=2)               # [B, C]

        # Get distances of each feature with its center: D(f_i, c_yi)
        target_diff = dist_mat.gather(1, targets.view(-1, 1))       # [B, 1]

        # In order to get minimum distance, here add the maximum distance at target column
        max_dist = dist_mat.max(dim=1, keepdim=True).values         # [B, 1]
        dist_mat.scatter_add_(1, target_diff.view(-1, 1), max_dist)

        # Get the minimum distances min( D(f_i, c_j) ),  j != y_i
        min_other_diff, min_idxs = dist_mat.min(dim=1)              # [B, 1]

        # Triplet-Center Loss
        losses = target_diff.squeeze() + margin - min_other_diff
        losses = torch.clamp(losses, min=0.0)

        # gradient mask
        mask = torch.gt(losses, 0.0)

        return torch.mean(losses)

    @staticmethod
    def backward(ctx, grad_outputs):



        return _, _, None, None
