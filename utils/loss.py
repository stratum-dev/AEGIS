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


import torch
import torch.nn.functional as F


def soft_f1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-7,
    reduction: str = "macro",
    detach_denominator: bool = True,
    ignore_empty_classes: bool = True,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.size(1)

    y_true = F.one_hot(targets, num_classes=num_classes).float()

    tp = (probs * y_true).sum(dim=0)
    fp = (probs * (1 - y_true)).sum(dim=0)
    fn = ((1 - probs) * y_true).sum(dim=0)

    if reduction == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        denom = 2 * tp_sum + fp_sum + fn_sum
        if detach_denominator:
            denom = denom.detach()
        f1 = (2 * tp_sum) / (denom + epsilon)

    else:  # macro
        denom = 2 * tp + fp + fn
        if detach_denominator:
            denom = denom.detach()

        f1_per_class = (2 * tp) / (denom + epsilon)

        if ignore_empty_classes:
            mask = y_true.sum(dim=0) > 0
            if mask.any():
                f1_per_class = f1_per_class[mask]
            else:
                return torch.tensor(0.0, device=logits.device)

        f1 = f1_per_class.mean()

    return 1 - f1
