import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Refer to paper [Supervised Contrastive Learning]
    (https://arxiv.org/pdf/2004.11362.pdf)

    Refer to paper [Deep Learning Face Representation by Joint Identification-Verification]
    (https://arxiv.org/pdf/1406.4773.pdf)
    """
    
    def __init__(self, margin: float, loss_weight: float):
        super(ContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.register_buffer("margin", torch.tensor([margin], dtype=torch.float32))

    def forward(self, feats, targets):
        dist_mat = _calc_pairwise_distance(feats)
        mask_anc_pos = _get_contrastive_pair(targets)
        mask_anc_neg = torch.logical_not(mask_anc_pos)
        mask_anc_pos = mask_anc_pos.to(dtype=torch.float32)
        mask_anc_neg = mask_anc_neg.to(dtype=torch.float32)

        loss_anc_pos = mask_anc_pos * dist_mat
        loss_anc_neg = self.margin - mask_anc_neg * dist_mat
        loss_anc_neg = torch.clamp(loss_anc_neg, min=0.0)
        loss = torch.mean(loss_anc_pos + loss_anc_neg)
        loss = self.loss_weight * loss
        return loss


def _calc_pairwise_distance(embeddings):
    # Dot product between all embeddings
    dot_product = torch.matmul(embeddings, embeddings.T)  # [B, B]

    # Get the squared L2-Norm for each embedding
    squared_norm = torch.diag(dot_product, 0)  # [B]

    # ||A - B||^2 = A^2 - 2 * A * B + B^2
    dist_mat = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
    return dist_mat


def _get_contrastive_pair(labels):
    idx_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    idx_not_equal = torch.logical_not(idx_equal)

    lbl_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

    mask = idx_not_equal & lbl_equal
    return mask