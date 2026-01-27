import torch
from torch.functional import F


def kappa_loss(logits, classes):
    return F.cross_entropy(logits, classes)


def prototype_loss(
    embeddings: torch.Tensor,  # (N, D)
    classes: torch.Tensor,  # (N,)
    prototypes: torch.Tensor,  # (C, D)
    temperature: float = 1.0,
):

    embeddings = F.normalize(embeddings, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    # logits: (N, C)
    logits = torch.matmul(embeddings, prototypes.t()) / temperature

    loss = F.cross_entropy(logits, classes)
    return loss
