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


def proto_loss_vmf(prototypes, kappas, eps=1e-6):

    device = prototypes.device
    C, d = prototypes.shape

    kappas = kappas.clamp(min=eps)

    # alpha_c = sqrt((d-1)/kappa_c)
    alpha = torch.sqrt((d - 1) / kappas)  # [C]

    # cosine similarity matrix
    G = prototypes @ prototypes.T  # [C, C]

    # build alpha_i + alpha_j
    alpha_i = alpha.view(C, 1)
    alpha_j = alpha.view(1, C)
    alpha_sum = alpha_i + alpha_j  # [C, C]

    # allowed max cosine
    cos_bound = torch.cos(alpha_sum)

    # violation: mu_i^T mu_j - cos(alpha_i + alpha_j)
    violation = G - cos_bound

    # remove diagonal
    eye = torch.eye(C, device=device).bool()
    violation = violation.masked_fill(eye, 0.0)

    # hinge
    loss = torch.relu(violation)

    # only count upper triangle (avoid double counting)
    loss = loss.triu(diagonal=1)

    return loss.mean()
