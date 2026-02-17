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
