import numpy as np
from sklearn.metrics import calinski_harabasz_score
import torch
from torch.functional import F


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


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize tensor along last dimension"""
    return F.normalize(x, p=2, dim=-1)


def cosine_similarity_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between x and y (both normalized)"""
    return torch.sum(x * y, dim=-1)


def estimate_vmf_concentration(embeddings: torch.Tensor) -> float:
    """
    Estimate vMF concentration parameter κ from normalized embeddings.
    embeddings: (N, D)
    """
    if embeddings.size(0) == 0:
        return 0.0
    mean_vec = torch.mean(embeddings, dim=0)
    r = torch.norm(mean_vec).item()
    d = embeddings.size(1)
    if r >= 1.0:
        r = 0.999999
    if r <= 0:
        return 0.0
    kappa = r * (d - r * r) / (1 - r * r)
    return max(kappa, 1e-6)


def compute_ch_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz Index"""
    if len(np.unique(labels)) <= 1:
        return 0.0
    return calinski_harabasz_score(embeddings, labels)


def compute_dunn_index(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Dunn Index: min_inter_cluster_distance / max_intra_cluster_diameter"""
    from scipy.spatial.distance import pdist, squareform

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Compute pairwise distances
    distances = squareform(pdist(embeddings, metric="euclidean"))

    min_inter = float("inf")
    max_intra = 0.0

    for i, label_i in enumerate(unique_labels):
        mask_i = labels == label_i
        cluster_i = embeddings[mask_i]

        # Intra-cluster diameter (max distance within cluster)
        if len(cluster_i) > 1:
            intra_dists = pdist(cluster_i, metric="euclidean")
            max_intra = max(max_intra, np.max(intra_dists))
        else:
            # Single point → diameter = 0
            pass

        # Inter-cluster distance to other clusters (min distance between any two points)
        for label_j in unique_labels[i + 1 :]:
            mask_j = labels == label_j
            cluster_j = embeddings[mask_j]

            # Compute min distance between cluster_i and cluster_j
            inter_dists = distances[np.ix_(mask_i, mask_j)]
            min_inter = min(min_inter, np.min(inter_dists))

    if max_intra == 0 or min_inter == float("inf"):
        return 0.0
    return min_inter / max_intra


def compute_class_separation_index(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Class Separation Index = min_{i≠j} ||μ_i - μ_j||_2 / max_k (diameter of cluster k)
    """
    from scipy.spatial.distance import pdist

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Compute centroids and diameters
    centroids = []
    diameters = []

    for label in unique_labels:
        cluster = embeddings[labels == label]
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)

        if len(cluster) > 1:
            intra_dists = pdist(cluster, metric="euclidean")
            diameter = np.max(intra_dists)
        else:
            diameter = 0.0
        diameters.append(diameter)

    centroids = np.array(centroids)
    max_diameter = np.max(diameters)

    # Min distance between any two centroids
    if len(centroids) > 1:
        inter_centroid_dists = pdist(centroids, metric="euclidean")
        min_centroid_dist = np.min(inter_centroid_dists)
    else:
        min_centroid_dist = 0.0

    if max_diameter == 0:
        return float("inf") if min_centroid_dist > 0 else 0.0
    return min_centroid_dist / max_diameter
