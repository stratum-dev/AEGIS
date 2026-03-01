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


def evaluate_etf_proximity(prototypes: torch.Tensor):
    """
    prototypes: [C, D], assumed L2-normalized
    return: dict with ETF diagnostics
    """

    C = prototypes.size(0)
    device = prototypes.device

    # Gram matrix
    G = prototypes @ prototypes.T

    eye = torch.eye(C, device=device).bool()
    off = G[~eye]

    # === theoretical simplex value ===
    target = -1.0 / (C - 1)

    # === metrics ===
    mean_cos = off.mean().item()
    std_cos = off.std().item()
    max_dev = (off - target).abs().max().item()
    mse = ((off - target) ** 2).mean().item()

    # === frame potential ===
    # FP = sum_ij <wi,wj>^2
    fp = (G**2).sum().item()

    # ETF minimum FP = C^2 / D  (for unit vectors)
    D = prototypes.size(1)
    fp_opt = (C**2) / D

    fp_ratio = fp / fp_opt

    # === collapse score (prototype norm sanity) ===
    norms = prototypes.norm(dim=1)
    norm_std = norms.std().item()

    # === heuristic ETF score (0~1, higher better) ===
    score = torch.exp(-10 * torch.tensor(mse)).item()

    return {
        "C": C,
        "target_cos": target,
        "mean_cos": mean_cos,
        "std_cos": std_cos,
        "max_abs_dev": max_dev,
        "mse_to_simplex": mse,
        "frame_potential": fp,
        "fp_optimal": fp_opt,
        "fp_ratio": fp_ratio,
        "norm_std": norm_std,
        "etf_score": score,
    }


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


import numpy as np


def l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


# 1️⃣ Mean Resultant Length
def mean_resultant_length(X):
    """
    X: (N, d), assumed to be unit vectors
    return: scalar R
    """
    mean_vec = np.mean(X, axis=0)
    R = np.linalg.norm(mean_vec)
    return R


# 2️⃣ Angular Variance
def angular_variance(X):
    """
    Var_theta = 1 - R
    """
    R = mean_resultant_length(X)
    return 1.0 - R


# 3️⃣ Pairwise Angular Dispersion
def pairwise_angular_dispersion(X):
    """
    Compute mean and std of (1 - cosine)
    """
    # cosine similarity matrix
    cos_sim = X @ X.T

    # remove diagonal
    N = X.shape[0]
    mask = ~np.eye(N, dtype=bool)
    cos_vals = cos_sim[mask]

    dispersion = 1 - cos_vals
    return dispersion.mean(), dispersion.std()


# 4️⃣ Geodesic Variance
def geodesic_variance(X):
    """
    Var_geo = E[theta^2]
    theta = arccos(cosine)
    """
    cos_sim = X @ X.T

    N = X.shape[0]
    mask = ~np.eye(N, dtype=bool)

    cos_vals = np.clip(cos_sim[mask], -1.0, 1.0)
    theta = np.arccos(cos_vals)

    return np.mean(theta**2)


def between_class_angular_margin(X, y):
    """
    计算类中心之间的最小角距离（margin）
    以及平均类间角距离

    return:
        min_margin (radians)
        mean_margin (radians)
    """
    classes = np.unique(y)
    if len(classes) == 0:
        return 0.0, 0.0
    centroids = []

    for c in classes:
        Xc = X[y == c]
        centroid = np.mean(Xc, axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)

    centroids = np.stack(centroids)

    # 计算类中心之间的角度
    cos_sim = centroids @ centroids.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    K = len(classes)
    mask = ~np.eye(K, dtype=bool)
    angles = np.arccos(cos_sim[mask])

    return angles.min(), angles.mean()


def spherical_silhouette_score(X, y):
    """
    使用角距离 (arccos) 作为距离度量
    返回整体 silhouette 均值
    """
    N = X.shape[0]
    cos_sim = X @ X.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    theta = np.arccos(cos_sim)  # geodesic distance matrix

    scores = []

    for i in range(N):
        same_mask = y == y[i]
        other_mask = y != y[i]

        same_mask[i] = False  # 排除自身

        if np.sum(same_mask) == 0:
            continue

        a = np.mean(theta[i, same_mask])

        b = np.inf
        for c in np.unique(y[other_mask]):
            cluster_mask = y == c
            b = min(b, np.mean(theta[i, cluster_mask]))

        s = (b - a) / max(a, b)
        scores.append(s)

    return np.mean(scores)


def spherical_distance_matrix(X):
    cos_sim = X @ X.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim)


def spherical_silhouette_score(X, labels):
    N = len(X)
    theta = spherical_distance_matrix(X)
    labels = np.array(labels)

    scores = []
    for i in range(N):
        same = labels == labels[i]
        other = labels != labels[i]
        same[i] = False

        if np.sum(same) == 0:
            continue

        a = np.mean(theta[i, same])

        b = np.inf
        for c in np.unique(labels[other]):
            mask = labels == c
            b = min(b, np.mean(theta[i, mask]))

        s = (b - a) / max(a, b)
        scores.append(s)

    return np.mean(scores)


def spherical_davies_bouldin(X, labels):
    labels = np.array(labels)
    classes = np.unique(labels)

    centroids = []
    scatters = []

    for c in classes:
        Xc = X[labels == c]
        centroid = np.mean(Xc, axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)

        cos_sim = np.clip(Xc @ centroid, -1.0, 1.0)
        angles = np.arccos(cos_sim)
        scatters.append(np.mean(angles))

    centroids = np.stack(centroids)
    scatters = np.array(scatters)

    K = len(classes)
    db_values = []

    for i in range(K):
        max_ratio = -np.inf
        for j in range(K):
            if i == j:
                continue

            cos_sim = np.clip(np.dot(centroids[i], centroids[j]), -1.0, 1.0)
            dist = np.arccos(cos_sim)

            ratio = (scatters[i] + scatters[j]) / dist
            max_ratio = max(max_ratio, ratio)

        db_values.append(max_ratio)

    return np.mean(db_values)
