import numpy as np
from sklearn.metrics import calinski_harabasz_score
import torch
from torch.functional import F
from scipy.spatial import SphericalVoronoi


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


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


def pairwise_angle(x):
    """
    计算两两向量夹角 (弧度)
    """
    cos = x @ x.T
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize tensor along last dimension"""
    return F.normalize(x, p=2, dim=-1)


def cosine_similarity_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between x and y (both normalized)"""
    return torch.sum(x * y, dim=-1)


def compute_abd(geo_prototypes, class_means):
    """
    ABD: Angular Boundary Deviation
    """
    W = normalize(geo_prototypes)
    M = normalize(class_means)

    proto_angles = pairwise_angle(W)
    mean_angles = pairwise_angle(M)

    C = W.shape[0]

    abd = 0
    count = 0

    for i in range(C):
        for j in range(i + 1, C):
            abd += abs(proto_angles[i, j] - mean_angles[i, j])
            count += 1

    return abd / count


def compute_pcr(geo_prototypes, class_means):
    """
    Prototype Collapse Ratio
    """
    W = normalize(geo_prototypes)
    M = normalize(class_means)

    proto_angles = pairwise_angle(W)
    mean_angles = pairwise_angle(M)

    C = W.shape[0]

    ratios = []

    for i in range(C):
        for j in range(i + 1, C):
            theta_p = proto_angles[i, j]
            theta_m = mean_angles[i, j]

            ratios.append(abs(theta_p - theta_m) / theta_p)

    return np.mean(ratios)


def compute_collapse_metrics(geo_prototypes, class_means):

    abd = compute_abd(geo_prototypes, class_means)
    pcr = compute_pcr(geo_prototypes, class_means)

    widths = compute_effective_width(class_means)
    W = normalize(geo_prototypes)
    M = normalize(class_means)

    dist, proto_area, data_area = spherical_voronoi_distortion(W, M)
    area_diff = proto_area - data_area

    svd = dist

    return {
        "Boundary Deviation": abd,
        "Prototype Collapse Ratio": pcr,
        "Effective Width Average": widths.mean(),
        "Effective Width Std": widths.std(),
        "Prototype Areas": proto_area,
        "Data Areas": data_area,
        "Area Diff": area_diff,
        "Spherical Voronoi Distortion": svd,
    }


def compute_effective_width(class_geo_prototypes):
    """
    每个类别有效角宽
    """
    M = normalize(class_geo_prototypes)
    angles = pairwise_angle(M)

    C = M.shape[0]

    widths = []

    for i in range(C):
        others = [angles[i, j] for j in range(C) if j != i]
        widths.append(np.mean(others))

    return np.array(widths)


def spherical_cell_areas(points):
    """
    计算单位球面 Voronoi cell 面积
    """
    sv = SphericalVoronoi(points)
    sv.sort_vertices_of_regions()

    areas = []

    for region in sv.regions:
        vertices = sv.vertices[region]

        area = SphericalVoronoi.calculate_areas(SphericalVoronoi(vertices)).sum()

        areas.append(area)

    return np.array(areas)


def spherical_voronoi_distortion(geo_prototypes, class_means):

    W = normalize(geo_prototypes)
    M = normalize(class_means)

    proto_areas = spherical_cell_areas(W)
    data_areas = spherical_cell_areas(M)

    proto_areas /= proto_areas.sum()
    data_areas /= data_areas.sum()

    distortion = np.mean(np.abs(proto_areas - data_areas))

    return distortion, proto_areas, data_areas


def evaluate_etf_proximity(geo_prototypes: torch.Tensor):
    """
    Evaluate how close a set of prototypes is to an Equiangular Tight Frame (ETF).

    Args:
        geo_prototypes : Tensor [C, D]
            Class prototypes / centroids.
            Should already be L2-normalized.

    Returns:
        dict containing ETF diagnostics.
    """

    C = geo_prototypes.size(0)
    D = geo_prototypes.size(1)
    device = geo_prototypes.device

    # ------------------------------------------------------------
    # 1️⃣ Gram Matrix
    # ------------------------------------------------------------
    # G_ij = <w_i, w_j>
    # measures cosine similarity between every pair of prototypes
    G = geo_prototypes @ geo_prototypes.T

    # remove diagonal (self similarity = 1)
    eye = torch.eye(C, device=device).bool()
    off_diag_cosines = G[~eye]

    # ------------------------------------------------------------
    # 2️⃣ Theoretical ETF cosine
    # ------------------------------------------------------------
    # In a perfect ETF:
    # cos(w_i, w_j) = -1/(C-1)
    target_simplex_cosine = -1.0 / (C - 1)

    # ------------------------------------------------------------
    # 3️⃣ Pairwise cosine diagnostics
    # ------------------------------------------------------------
    # mean cosine between different class prototypes
    mean_interclass_cosine = off_diag_cosines.mean().item()
    distance_to_mean_intraclass_cosine = abs(
        mean_interclass_cosine - target_simplex_cosine
    )

    # std of those cosines
    # small std means angles are nearly equal → closer to simplex
    std_interclass_cosine = off_diag_cosines.std().item()

    # maximum deviation from ideal ETF cosine
    max_abs_cosine_deviation = (
        (off_diag_cosines - target_simplex_cosine).abs().max().item()
    )

    # mean squared error to theoretical simplex cosine
    mse_to_simplex_structure = (
        ((off_diag_cosines - target_simplex_cosine) ** 2).mean().item()
    )

    # ------------------------------------------------------------
    # 4️⃣ Frame Potential
    # ------------------------------------------------------------
    # FP = Σ_ij <w_i,w_j>^2
    # ETF minimizes frame potential for unit vectors
    frame_potential = (G**2).sum().item()

    # theoretical minimum FP for ETF
    optimal_frame_potential = (C**2) / D

    # ratio to optimum (1.0 means perfect ETF)
    frame_potential_ratio = frame_potential / optimal_frame_potential

    # ------------------------------------------------------------
    # 5️⃣ Prototype norm sanity check
    # ------------------------------------------------------------
    # if prototypes are normalized correctly,
    # norms should all be ≈1
    prototype_norm_std = geo_prototypes.norm(dim=1).std().item()

    # ------------------------------------------------------------
    # 6️⃣ Heuristic ETF score (0~1)
    # ------------------------------------------------------------
    # exponential transform of MSE
    # closer to ETF → score closer to 1
    etf_alignment_score = torch.exp(-10 * torch.tensor(mse_to_simplex_structure)).item()

    return {
        # pairwise cosine diagnostics
        "distance_to_mean_interclass_cosine": distance_to_mean_intraclass_cosine,
        "std_interclass_cosine": std_interclass_cosine,
        "max_abs_cosine_deviation": max_abs_cosine_deviation,
        "mse_to_simplex_structure": mse_to_simplex_structure,
        # frame potential diagnostics
        "frame_potential": frame_potential,
        "optimal_frame_potential": optimal_frame_potential,
        "frame_potential_ratio": frame_potential_ratio,
        # normalization sanity
        "prototype_norm_std": prototype_norm_std,
        # overall heuristic score
        "etf_alignment_score": etf_alignment_score,
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


def prototype_alignment(weight_prototypes, geo_prototypes):
    """
    计算 prototypes 的角度对齐程度

    Args:
        weight_prototypes : (K, D)
        geo_prototypes    : (K, D)

    Returns:
        mean_angle (radians)
        per_class_angles
    """

    # ---- 1️⃣ L2 normalize ----
    W = weight_prototypes / np.linalg.norm(weight_prototypes, axis=1, keepdims=True)
    G = geo_prototypes / np.linalg.norm(geo_prototypes, axis=1, keepdims=True)

    # ---- 2️⃣ cosine similarity ----
    cos_vals = np.sum(W * G, axis=1)
    cos_vals = np.clip(cos_vals, -1.0, 1.0)

    # ---- 3️⃣ angle ----
    angles = np.arccos(cos_vals)

    return angles.mean()


def between_class_angular_margin(geo_prototypes):
    """
    计算类中心之间的最小角距离（margin）
    以及平均类间角距离

    return:
        min_margin (radians)
        mean_margin (radians)
    """

    # 计算类中心之间的角度
    cos_sim = geo_prototypes @ geo_prototypes.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    K = geo_prototypes.shape[0]
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
