import torch
from torch.functional import F


def kappa_loss(logits, classes):
    return F.cross_entropy(logits, classes)


def prototype_consistency_loss(
    weight_proto: torch.Tensor,
    geo_proto: torch.Tensor,
):
    weight_proto = F.normalize(weight_proto, dim=1)
    geo_proto = F.normalize(geo_proto, dim=1)

    # cosine similarity per class
    cos_sim = torch.sum(weight_proto * geo_proto, dim=1)
    loss = 1.0 - cos_sim.mean()
    return loss
