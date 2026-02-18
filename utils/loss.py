import torch
from torch.functional import F


def kappa_loss(logits, classes):
    return F.cross_entropy(logits, classes)


def prototype_consistency_loss(
    weight_proto: torch.Tensor,
    avg_proto: torch.Tensor,
):
    weight_proto = F.normalize(weight_proto, dim=1)
    avg_proto = F.normalize(avg_proto, dim=1)

    # cosine similarity per class
    cos_sim = torch.sum(weight_proto * avg_proto, dim=1)
    loss = 1.0 - cos_sim.mean()
    return loss


def prototype_alignment_loss(
    embeddings: torch.Tensor,  # [B, d] 已 normalize
    labels: torch.Tensor,  # [B]
    weight_proto: torch.Tensor,  # [C, d] 已 normalize + detach
):
    """
    PPC loss: pull embeddings toward frozen classifier prototypes.

    embeddings: normalized features
    labels: ground truth class indices
    weight_proto: frozen classifier weights (normalized)
    """

    # 取每个样本对应的 prototype
    proto_per_sample = weight_proto[labels]  # [B, d]

    # cosine pull
    loss = 1.0 - F.cosine_similarity(embeddings, proto_per_sample, dim=1).mean()

    return loss
