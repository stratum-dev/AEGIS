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


def pcgrad_backward(
    losses: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler = None,
):
    """
    Perform PCGrad backward pass with optional AMP support.

    Args:
        losses (dict): Dictionary of loss tensors, e.g., {"kappa": loss1, "ppc": loss2}
        optimizer (torch.optim.Optimizer): Optimizer whose parameters will be updated
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision training
    """
    params = [p for p in optimizer.param_groups[0]["params"] if p.requires_grad]
    grads = {}
    names = list(losses.keys())

    # Compute gradients for each loss
    for i, (name, loss) in enumerate(losses.items()):
        optimizer.zero_grad(set_to_none=True)
        retain_graph = i < len(names) - 1  # Only retain graph if not the last loss
        if scaler is not None:
            scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)
        grads[name] = [None if p.grad is None else p.grad.clone() for p in params]

    # PCGrad projection: resolve conflicting gradients
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            g_i = grads[names[i]]
            g_j = grads[names[j]]

            dot = sum(
                (gi * gj).sum()
                for gi, gj in zip(g_i, g_j)
                if gi is not None and gj is not None
            )
            if dot < 0:
                norm = sum((gj * gj).sum() for gj in g_j if gj is not None) + 1e-8
                coeff = dot / norm
                for k in range(len(g_i)):
                    if g_i[k] is not None and g_j[k] is not None:
                        g_i[k] -= coeff * g_j[k]

    # Write back merged gradients
    optimizer.zero_grad(set_to_none=True)
    for p_idx, p in enumerate(params):
        grad_list = [
            grads[name][p_idx] for name in grads if grads[name][p_idx] is not None
        ]
        if grad_list:
            p.grad = sum(grad_list)
        else:
            p.grad = None
