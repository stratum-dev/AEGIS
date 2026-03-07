import warnings
from typing import Dict, List, Tuple, Any
import numpy as np
from torch.functional import F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
import torch

warnings.filterwarnings("ignore")


class ClassificationMetricCalculator:
    """指标计算器类"""

    @staticmethod
    def hierarchical_decision(
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> List[int]:
        # sims = torch.mm(embeddings, prototypes.T)
        # pred_indicies = sims.argmax(dim=1)

        # return pred_indicies.tolist()
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(prototypes, dim=1)
        logits = torch.matmul(emb_norm, w_norm.t())
        return torch.argmax(logits, dim=1).tolist()

    @staticmethod
    def calculate_l1_metrics(
        y_true_binary: np.ndarray, y_pred_binary: np.ndarray
    ) -> Dict[str, float]:
        """计算二分类指标"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(
            y_true_binary, y_pred_binary, labels=[False, True]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "mcc": float(mcc),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    @staticmethod
    def calculate_l2_metrics(
        all_pred_class_indices: List[int],
        all_true_class_keys: List[Tuple[bool, str]],
        idx_to_class,
    ) -> Dict[str, Any]:
        y_true_cwe: List[str] = []
        y_pred_cwe: List[str] = []

        for pred_idx, (true_label, true_cwe) in zip(
            all_pred_class_indices, all_true_class_keys
        ):
            if not true_label:
                continue  # Oracle: only GT vulnerable samples

            y_true_cwe.append(true_cwe)

            pred_label, pred_cwe = idx_to_class[pred_idx]

            if not pred_label:
                # predicted as non-vul → CWE prediction failure
                y_pred_cwe.append("__UNKNOWN__")
            else:
                y_pred_cwe.append(pred_cwe)

        unique_cwes = sorted(set(y_true_cwe))

        if not unique_cwes:
            return {
                "per_class": {},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0},
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # =========================
        # Precision / Recall / F1
        # =========================
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            average="micro",
            zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            average="macro",
            zero_division=0,
        )

        per_class_report = classification_report(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            zero_division=0,
            output_dict=True,
        )

        # =========================
        # Macro MCC (OVA, Oracle CWE)
        # =========================
        per_class_mcc = {}
        macro_mcc_list = []
        per_class_confusion = {}
        for cwe in unique_cwes:
            tp = fp = fn = tn = 0

            for yt, yp in zip(y_true_cwe, y_pred_cwe):
                if yt == cwe:
                    if yp == cwe:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if yp == cwe:
                        fp += 1
                    else:
                        tn += 1

            per_class_confusion[cwe] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }

        for cwe in unique_cwes:
            y_true_ova = [1 if y == cwe else 0 for y in y_true_cwe]
            y_pred_ova = [1 if y == cwe else 0 for y in y_pred_cwe]

            mcc_val = matthews_corrcoef(y_true_ova, y_pred_ova)
            mcc_val = 0.0 if np.isnan(mcc_val) else float(mcc_val)

            per_class_mcc[cwe] = mcc_val
            macro_mcc_list.append(mcc_val)

        macro_mcc = float(np.mean(macro_mcc_list))

        per_class_metrics = {
            cwe: {
                "precision": per_class_report[cwe]["precision"],
                "recall": per_class_report[cwe]["recall"],
                "f1-score": per_class_report[cwe]["f1-score"],
                "support": per_class_report[cwe]["support"],
                "mcc": per_class_mcc[cwe],
                **per_class_confusion[cwe],
            }
            for cwe in unique_cwes
        }

        end2end_correct = 0
        total = len(all_true_class_keys)
        for pred_idx, (true_label, true_cwe) in zip(
            all_pred_class_indices, all_true_class_keys
        ):
            # =========================
            # GT: Non-vulnerable
            # =========================
            if not true_label:
                pred_label, _ = idx_to_class[pred_idx]
                if not pred_label:
                    end2end_correct += 1

            # =========================
            # GT: Vulnerable
            # =========================

            pred_label, pred_cwe = idx_to_class[pred_idx]

            if not pred_label:
                continue  # predicted non-vul → wrong

            if pred_cwe == true_cwe:
                end2end_correct += 1

        return {
            "per_class": per_class_metrics,
            "macro": {
                "precision": float(macro_p),
                "recall": float(macro_r),
                "f1": float(macro_f1),
                "mcc": float(macro_mcc),
            },
            "micro": {
                "precision": float(micro_p),
                "recall": float(micro_r),
                "f1": float(micro_f1),
            },
            "hier_acc": end2end_correct / total if total > 0 else 0.0,
        }


class ETFMetricCalculator:
    @staticmethod
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
        etf_alignment_score = torch.exp(
            -10 * torch.tensor(mse_to_simplex_structure)
        ).item()

        return {
            "Distance to Mean Interclass Cosine": distance_to_mean_intraclass_cosine,
            "Std Interclass Cosine": std_interclass_cosine,
            "Max Abs Cosine Deviation": max_abs_cosine_deviation,
            "MSE to Simplex Structure": mse_to_simplex_structure,
            "Frame Potential": frame_potential,
            "Optimal Frame Potential": optimal_frame_potential,
            "Frame Potential Ratio": frame_potential_ratio,
            "Prototype Norm Std": prototype_norm_std,
            "ETF Alignment Score": etf_alignment_score,
        }


class DistortionMetricCalculator:
    @staticmethod
    def compute_distortion_metrics(embs, class_indices, avg_prototypes):
        """
        embs: np.array of length N, each element is a torch tensor of shape (1,D)
        class_indices: np.array of shape [N,]
        avg_prototypes: torch tensor of shape [C,D], L2 normalized
        """
        # 1. 将 embs 转成 torch tensor [N,D]
        embs_tensor = torch.stack(
            [
                (
                    e.squeeze(0)
                    if isinstance(e, torch.Tensor)
                    else torch.tensor(e).squeeze(0)
                )
                for e in embs
            ],
            dim=0,
        )  # [N,D]
        class_indices = torch.tensor(class_indices)  # [N,]

        C = avg_prototypes.shape[0]

        # 2. 计算余弦相似度
        similarities = embs_tensor @ avg_prototypes.T  # [N,C]

        # 3. argmax 决策
        pred = similarities.argmax(dim=1)  # [N,]

        # 4. 初始化统计量
        intrusion_list = []
        collapse_list = []

        # 5. 遍历类别对计算 intrusion / collapse
        for i in range(C):
            for j in range(i + 1, C):
                mask_i = class_indices == i
                if mask_i.sum() > 0:
                    intrusion = (pred[mask_i] == j).float().mean()
                    intrusion_list.append(intrusion)
                mask_j = class_indices == j
                if mask_j.sum() > 0:
                    collapse = (pred[mask_j] == i).float().mean()
                    collapse_list.append(collapse)

        avg_intrusion = (
            torch.tensor(intrusion_list).mean().item() if intrusion_list else 0.0
        )
        avg_collapse = (
            torch.tensor(collapse_list).mean().item() if collapse_list else 0.0
        )
        avg_distortion = avg_intrusion + avg_collapse

        return {
            "Decision Intrusion Area": avg_intrusion,
            "Decision Collapse Area": avg_collapse,
            "Distortion Area": avg_distortion,
        }


class GeometryMetricCalculator:
    @staticmethod
    def compute_ch_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calinski-Harabasz Index"""
        if len(np.unique(labels)) <= 1:
            return 0.0
        return calinski_harabasz_score(embeddings, labels)

    @staticmethod
    def compute_class_separation_index(
        embeddings: np.ndarray, labels: np.ndarray
    ) -> float:
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

    @staticmethod
    def mean_resultant_length(X):
        """
        X: (N, d), assumed to be unit vectors
        return: scalar R
        """
        mean_vec = np.mean(X, axis=0)
        R = np.linalg.norm(mean_vec)
        return R

    @staticmethod
    def angular_variance(X):
        """
        Var_theta = 1 - R
        """
        R = GeometryMetricCalculator.mean_resultant_length(X)
        return 1.0 - R

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def calculate_geometry_metrics(geo_prototypes, weight_prototypes):
        geo_prototype_mrl = GeometryMetricCalculator.mean_resultant_length(
            geo_prototypes
        )
        geo_prototype_angular_var = GeometryMetricCalculator.angular_variance(
            geo_prototypes
        )
        geo_prototype_pariwise_dispersion = (
            GeometryMetricCalculator.pairwise_angular_dispersion(geo_prototypes)
        )
        geo_prototype_geodesic_variance = GeometryMetricCalculator.geodesic_variance(
            geo_prototypes
        )
        geo_protptype_bcm = GeometryMetricCalculator.between_class_angular_margin(
            geo_prototypes
        )
        geo_weight_protptype_alignment = GeometryMetricCalculator.prototype_alignment(
            geo_prototypes, weight_prototypes
        )

        return {
            "MRL": geo_prototype_mrl,
            "Geo-Weight Prototype alignment": geo_weight_protptype_alignment,
            "Angular Var": geo_prototype_angular_var,
            "Pairwise Disperation Average": geo_prototype_pariwise_dispersion[0],
            "Pairwise Disperation Std": geo_prototype_pariwise_dispersion[1],
            "Geodesic Variance": geo_prototype_geodesic_variance,
            "Between Class Margin Minimun": geo_protptype_bcm[0],
            "Between Class Margin Average": geo_protptype_bcm[1],
        }


class ClusteringMetricCalculator:
    @staticmethod
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

    @staticmethod
    def spherical_distance_matrix(X):
        cos_sim = X @ X.T
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return np.arccos(cos_sim)

    @staticmethod
    def spherical_silhouette_score(X, labels):
        N = len(X)
        theta = ClusteringMetricCalculator.spherical_distance_matrix(X)
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

    @staticmethod
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

    @staticmethod
    def calculate_clustering_metrics(embeddings, true_labels, pred_labels):
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        silhouette_avg = (
            ClusteringMetricCalculator.spherical_silhouette_score(
                embeddings, pred_labels
            )
            if len(set(pred_labels)) > 1
            else 0
        )

        dbi_score = (
            ClusteringMetricCalculator.spherical_davies_bouldin(embeddings, pred_labels)
            if len(set(pred_labels)) > 1
            else float("inf")
        )

        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        ami_score = adjusted_mutual_info_score(true_labels, pred_labels)
        ari_score = adjusted_rand_score(true_labels, pred_labels)

        return {
            "SH": silhouette_avg,
            "NMI": nmi_score,
            "AMI": ami_score,
            "ARI": ari_score,
            "DBI": dbi_score,
        }
