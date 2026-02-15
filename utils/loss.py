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


def proto_loss_etf(prototypes):
    """
    prototypes: [C, d], already L2 normalized
    """

    device = prototypes.device
    C = prototypes.size(0)

    # Gram matrix
    G = prototypes @ prototypes.T   # [C,C]

    # ETF target off-diagonal value
    target = -1.0 / (C - 1)

    # mask diagonal
    eye = torch.eye(C, device=device).bool()

    # only off-diagonal entries
    off_diag = G[~eye]

    loss = ((off_diag - target) ** 2).mean()

    return loss