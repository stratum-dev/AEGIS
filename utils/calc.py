import torch
import numpy as np
import torch.nn.functional as F


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize along last dimension"""
    return F.normalize(x, p=2, dim=-1)


def geometric_median(
    X: torch.Tensor, eps: float = 1e-6, max_iter: int = 100
) -> torch.Tensor:
    y = X.mean(dim=0)
    for _ in range(max_iter):
        dist = torch.norm(X - y, dim=1)
        mask = dist > eps
        if not mask.any():
            break
        inv_dist = 1.0 / dist[mask]
        y_new = (X[mask] * inv_dist[:, None]).sum(dim=0) / inv_dist.sum()
        if torch.norm(y - y_new) < eps:
            break
        y = y_new
    return y


def estimate_vmf_concentration(
    embeddings: torch.Tensor, prototype: torch.Tensor
) -> float:

    if embeddings.size(0) == 0:
        return 0.0

    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    prototype = torch.nn.functional.normalize(prototype, dim=0)

    cos_sim = torch.matmul(embeddings, prototype)  # (N)
    r = torch.mean(cos_sim).item()

    d = embeddings.size(1)

    if r >= 1.0:
        r = 0.999999
    if r <= 0:
        return 0.0

    kappa = r * (d - r * r) / (1 - r * r)

    return max(kappa, 1e-6)
