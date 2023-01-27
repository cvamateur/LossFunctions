import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Refer to paper [Dimensionality Reduction by Learning an Invariant Mapping]
    (http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

    Refer to paper [Deep Learning Face Representation by Joint Identification-Verification]
    (https://arxiv.org/pdf/1406.4773.pdf)
    """
    
    def __init__(self, margin: float, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.register_buffer("margin", torch.tensor([margin], dtype=torch.float32))

    def forward(self, feats, targets):
        dist_mat = _calc_distance_matrix(feats, False)     # [B, B]

        mask = _get_contrastive_pair_mask(targets)
        mask = mask.to(dtype=torch.float32)

        # Loss for anchor-positive
        loss_anc_pos = mask * torch.square(dist_mat)

        # Loss for anchor-negative
        loss_anc_neg = torch.clamp(self.margin - dist_mat, min=0.0)
        loss_anc_neg = (1.0 - mask) * torch.square(loss_anc_neg)
        loss = loss_anc_pos + loss_anc_neg
        if self.reduction == "mean":
            return torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss



def _calc_distance_matrix(embeddings, squared: bool = True):
    # Dot product between all embeddings
    dot_product = torch.matmul(embeddings, embeddings.T)  # [B, B]

    # Get the squared L2-Norm for each embedding
    squared_norm = torch.diag(dot_product, 0)  # [B]

    # ||A - B||^2 = A^2 - 2 * A * B + B^2
    dist_mat = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
    dist_mat = dist_mat.clamp(min=0.0)

    if not squared:
        mask = torch.eq(dist_mat, 0.0).to(dtype=torch.float32)
        dist_mat = dist_mat + mask * 1e-8
        dist_mat = torch.sqrt(dist_mat)
        dist_mat = dist_mat * (1.0 - mask)

    return dist_mat


def _get_contrastive_pair_mask(labels):
    # Make sure i != j
    idx_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    idx_not_equal = torch.logical_not(idx_equal)

    # Make sure label[i] == label[j]
    lbl_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    return idx_not_equal & lbl_equal