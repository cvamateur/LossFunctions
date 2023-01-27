import torch
import torch.nn as nn

from torch.autograd import Function


class CenterLoss(nn.Module):
    """
    Refer to paper [A Discriminative Feature Learning Approach for Deep Face Recognition]
    (https://ydwen.github.io/papers/WenECCV16.pdf)

    This is an auxiliary loss that can be used togather with SoftmaxLoss.

    @Arguments:
    ----------
    feats (float32 Tensor[B, D]):
        Embeddings of batch images.

    labels (int64 Tensor[B]):
        A vector of target labels.

    @Usage:
    -------
    >> criterion0 = SoftmaxLoss()
    >> cirterion1 = CenterLoss(2, 10, loss_weight=0.01)
    >> feats = extractor(images)
    >> logits = classifier(feats)
    >> loss = criterion0(logits, labels) + criterion1(feats, labels)
    >> loss.backward()
    """

    def __init__(self, feats_dim: int, num_classes: int):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.zeros([num_classes, feats_dim], dtype=torch.float32))

    def forward(self, feats, targets):
        return _CenterLossFn.apply(feats, self.centers, targets)


class _CenterLossFn(Function):

    @staticmethod
    def forward(ctx, feats, centers, targets):
        diff = feats - centers[targets]
        loss = 0.5 * torch.mean(torch.sum(diff.square(), dim=1))
        ctx.save_for_backward(targets, centers, diff)
        return loss

    @staticmethod
    def backward(ctx, grad_outputs):
        targets, centers, diff = ctx.saved_tensors
        batch_size = diff.size(0)

        # gradient of dL/dX
        d_feats = grad_outputs * diff / batch_size

        # gradient of dL/dc
        weights = diff.new_ones(batch_size)
        counts = diff.new_ones([centers.size(0)]).scatter_add_(0, targets, weights)  # denominator
        d_centers = torch.zeros_like(centers)
        d_centers.scatter_add_(0, targets.unsqueeze(1).expand_as(diff), -1.0 * diff)  # numerator
        d_centers.div_(counts.view(-1, 1)) / batch_size
        return d_feats, d_centers, None
