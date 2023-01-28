import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function


class TripletCenterLoss(nn.Module):
    """
    Refer to paper [Triplet-Center Loss for Multi-View 3D Object Retrieval]
    (https://arxiv.org/pdf/1803.06189.pdf)
    """

    def __init__(self, feat_dims: int, num_classes: int, margin: float = 5.0):
        super(TripletCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.zeros([num_classes, feat_dims], dtype=torch.float32))
        self.register_buffer("margin", torch.tensor([margin], dtype=torch.float32))


        self._forward_impl =self._forward_impl_functional

    def _forward_impl_functional(self, feats, targets):
        return _TripletCenterLossFunc.apply(feats, self.centers, targets, self.margin)

    def _forward_impl(self, feats, targets):
        ...

    def forward(self, feats, targets):
        return self._forward_impl(feats, targets)


class _TripletCenterLossFunc(Function):

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
        dist_mat.scatter_add_(1, targets.view(-1, 1), max_dist)

        # Get the minimum distances min( D(f_i, c_j) ),  j != y_i
        min_other_diff, min_idxs = dist_mat.min(dim=1)              # [B, 1]

        # Triplet-Center Loss
        losses = target_diff.squeeze() + margin - min_other_diff
        losses = torch.clamp(losses, min=0.0)

        # gradient mask
        mask = torch.gt(losses, 0.0).to(dtype=torch.float32)
        ctx.save_for_backward(feats, targets, centers, min_idxs.squeeze(), mask)
        return torch.mean(losses)

    @staticmethod
    def backward(ctx, grad_outputs):
        feats, targets, centers, min_idx, mask = ctx.saved_tensors
        batch_size = targets.size(0)
        num_classes = centers.size(0)

        # fi - cj
        target_diff = mask.view(-1, 1) * (feats - centers[targets])
        min_idx_diff = mask.view(-1, 1) * (feats - centers[min_idx])

        # dL/df: Equation 6
        d_feats = grad_outputs * mask.view(-1, 1) * (centers[min_idx] - centers[targets]) / batch_size

        # dL/dc: Equation 7
        # The gradient is divided into two parts: pull and push
        d_centers_pull = torch.zeros_like(centers)
        d_centers_push = torch.zeros_like(centers)

        # 1. pull part
        weights = feats.new_ones([batch_size]) * mask
        counts = centers.new_ones([num_classes]).scatter_add_(0, targets, weights)  # denominator
        d_centers_pull.scatter_add_(0, targets.view(-1,1).expand_as(target_diff), target_diff)
        d_centers_pull.div_(counts.view(-1, 1))

        # 2. push part
        counts = centers.new_ones([num_classes]).scatter_add_(0, min_idx, weights)  # denominator
        d_centers_push.scatter_add_(0, min_idx.view(-1,1).expand_as(target_diff), min_idx_diff)
        d_centers_push.div_(counts.view(-1, 1))

        # Final dL/dc
        d_centers = grad_outputs * (d_centers_pull - d_centers_push) / batch_size
        return d_feats, d_centers, None, None
