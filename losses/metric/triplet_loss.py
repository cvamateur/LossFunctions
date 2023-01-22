import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """
    Refer to paper [FaceNet: A Unified Embedding for Face Recognition and Clustering]
    (https://arxiv.org/pdf/1503.03832.pdf)

    Refer to paper [In Defense of the Triplet Loss for Person Re-Identification]
    (https://arxiv.org/pdf/1703.07737.pdf)

    @Parameters:
    ------------
    margin (float):
        Margin in triplet-loss function: L = max(d(a,p) - d(a,n) + margin, 0)

    strategy(str):
        "batch_all":
            Select all the valid triplets, and average the loss on the hard and semi-hard triplets.
            NOTE:
                Easy triplets (those loss is 0) are not taken into account, as averaging on them
                would make the overall loss very small.

        "batch_hard":
            For each anchor, select the hardest positive (biggest distance d(a,p)) and
            the hardest negative among the batch.

    NOTE:
        Selecting the hardest negatives can in practice lead to bad local minima early on in training.
    """

    def __init__(self, margin: float, loss_weight: float, strategy: str = "batch_hard"):
        super(TripletLoss, self).__init__()
        self.register_buffer("margin", torch.tensor([margin], dtype=torch.float32))
        self.loss_weight = loss_weight
        if strategy == "batch_hard":
            self._forward_impl = self._forward_batch_hard
        elif strategy == "batch_all":
            self._forward_impl = self._forward_batch_all
        else:
            msg = "Strategy must be 'batch_hard' or 'batch_all'"
            raise RuntimeError(msg)

    def forward(self, feats, labels):
        loss = self._forward_impl(feats, labels)
        return self.loss_weight * loss

    # ---------------------------- Batch All -------------------------------
    def _forward_batch_all(self, feats, labels):
        dist_mat = _calc_pairwise_distance(feats)

        # Mask of valid triplets, affected by `strategy`
        mask = _get_mask_batch_all(labels)
        mask = mask.to(dtype=torch.float32, device=dist_mat.device)

        dist_anc_pos = dist_mat.unsqueeze(2)
        dist_anc_neg = dist_mat.unsqueeze(1)
        loss = dist_anc_pos - dist_anc_neg + self.margin        # [B, B, B]

        # Zeros all invalid triplets
        loss = torch.mul(loss, mask)

        # Remove negative losses
        loss = torch.clamp(loss, min=0.0)

        # Count number of positive triplets
        valid_triplets = torch.gt(loss, 1e-10).to(dtype=torch.float32)
        num_valid_pos_triplets = torch.sum(valid_triplets)

        # Reduce loss
        loss = torch.sum(loss) / (num_valid_pos_triplets + 1e-10)
        return loss

    def _forward_batch_hard(self, feats, labels):
        dist_mat = _calc_pairwise_distance(feats)

        # For each anchor, get the hardest positive and hardest negative
        mask_anc_pos, mask_anc_neg = _get_anc_pos_and_neg_mask(labels)
        mask_anc_pos = mask_anc_pos.to(dtype=torch.float32)
        mask_anc_neg = mask_anc_neg.to(dtype=torch.float32)

        # For each anchor, get the hardest positive
        # Put to 0 any element where (a, p) is not valid
        anc_pos_dist = dist_mat * mask_anc_pos
        hardest_pos_dist = torch.max(anc_pos_dist, dim=1).values    # [B]

        # For each anchor, get the hardest negative
        # Add the maximum value in each row to the invalid negatives
        max_dist = torch.max(dist_mat, dim=1, keepdim=True).values
        anc_neg_dist = dist_mat + max_dist * (1.0 - mask_anc_neg)
        hardest_neg_dist = torch.min(anc_neg_dist, dim=1).values    # [B]

        loss = hardest_pos_dist - hardest_neg_dist + self.margin
        loss = torch.mean(torch.clamp(loss, min=0.0))
        return loss


def _calc_pairwise_distance(embeddings):
    # Dot product between all embeddings
    dot_product = torch.matmul(embeddings, embeddings.T)      # [B, B]

    # Get the squared L2-Norm for each embedding
    squared_norm = torch.diag(dot_product, 0)                 # [B]

    # ||A - B||^2 = A^2 - 2 * A * B + B^2
    dist_mat = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
    return dist_mat

def _get_mask_batch_all(labels):
    idxs_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    idxs_not_equal = torch.logical_not(idxs_equal)

    # make sure i, j and k are distinct
    i_neq_j = idxs_not_equal.unsqueeze(2)
    i_neq_k = idxs_not_equal.unsqueeze(1)
    j_neq_k = idxs_not_equal.unsqueeze(0)
    valid_idxs = (i_neq_j & i_neq_k & j_neq_k)

    # make sure label[i] == label[j] and label[i] != label[k]
    lbl_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    lbl_i_eq_j = lbl_equal.unsqueeze(2)
    lbl_i_neq_k = lbl_equal.unsqueeze(1).logical_not()
    valid_lbls = lbl_i_eq_j & lbl_i_neq_k

    # Combine the two masks
    mask = valid_idxs & valid_lbls
    return mask

def _get_anc_pos_and_neg_mask(labels):
    # i and j are distinct
    idxs_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    idxs_not_equal = torch.logical_not(idxs_equal)

    # labels[i] != labels[j]
    lbl_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    lbl_not_equal = torch.logical_not(lbl_equal)
    mask_anc_pos = idxs_not_equal & lbl_not_equal
    mask_anc_neg = lbl_not_equal
    return mask_anc_pos, mask_anc_neg



