import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from umap import UMAP


class VisualizationHelper:
    """可视化辅助类"""

    @staticmethod
    def draw_plot_umap(
        embeddings: np.ndarray,
        class_keys: List[Tuple[bool, str]],
        output_dir: str,
        filename: str,
        title: str,
    ):
        """绘制UMAP降维图"""
        total_samples = len(class_keys)
        sampled_idx = np.arange(total_samples)

        sampled_embs = embeddings[sampled_idx]
        sampled_keys = [class_keys[i] for i in sampled_idx]

        reducer = UMAP(n_components=2, random_state=42)
        umap_embs = reducer.fit_transform(sampled_embs)

        labels = np.array([ck[0] for ck in sampled_keys])
        cwes = np.array([ck[1] for ck in sampled_keys])
        vuln_mask = labels.astype(bool)

        cwe_unique = sorted(set(cwes[vuln_mask])) if vuln_mask.any() else []
        cwe_to_color = {
            cwe: color
            for cwe, color in zip(
                cwe_unique, sns.color_palette("husl", len(cwe_unique))
            )
        }

        fig_width = max(10, 8 + 0.4 * (len(cwe_unique) + 1))
        plt.figure(figsize=(fig_width, 8))
        x, y = umap_embs[:, 0], umap_embs[:, 1]

        if not vuln_mask.all():
            non_vuln = ~vuln_mask
            plt.scatter(
                x[non_vuln],
                y[non_vuln],
                c="lightgray",
                marker="x",
                s=20,
                label="_nolegend_",
            )

        for cwe in cwe_unique:
            mask = (cwes == cwe) & vuln_mask
            if mask.any():
                plt.scatter(
                    x[mask], y[mask], c=[cwe_to_color[cwe]], marker="o", s=20, label=cwe
                )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, filename),
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def draw_prototype_similarity_matrix(
        prototypes: torch.Tensor,
        idx_to_class: Dict[int, Tuple[bool, str]],
        output_dir: str,
        filename: str,
        title: str,
    ):
        """绘制原型相似度热力图（仅下半三角，百分比显示）"""
        # 确保是 cosine similarity
        prototypes = torch.nn.functional.normalize(prototypes, dim=1)

        sim_matrix = torch.mm(prototypes, prototypes.t()).cpu().numpy()
        sim_percent = sim_matrix * 100.0

        class_labels = [f"{idx_to_class[i][1]}" for i in range(len(idx_to_class))]

        n = len(class_labels)
        size = max(6, n * 0.5)
        plt.figure(figsize=(size, size))

        mask = np.triu(np.ones_like(sim_percent, dtype=bool), k=1)

        ax = sns.heatmap(
            sim_percent,
            mask=mask,
            xticklabels=class_labels,
            yticklabels=class_labels,
            cmap="viridis",
            square=True,
            annot=True,
            fmt=".0f",
            vmin=-100.0,
            vmax=100.0,
            cbar_kws={"shrink": 0.8, "label": "Cosine Similarity (%)"},
        )

        # 百分比语义的 colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-100, -50, 0, 50, 100])
        cbar.set_ticklabels(["-100%", "-50%", "0%", "50%", "100%"])

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, filename),
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def draw_prototype_alignment_matrix(
        geo_prototypes: torch.Tensor,
        weight_prototypes: torch.Tensor,
        idx_to_class: Dict[int, Tuple[bool, str]],
        output_dir: str,
        filename: str,
        title: str,
        normalize: bool = True,
    ):
        """
        绘制 geo prototypes（纵轴）与 weight prototypes（横轴）的相似度热力图
        相似度定义为 cosine similarity（百分比显示）
        """

        assert (
            geo_prototypes.shape == weight_prototypes.shape
        ), "Geo prototypes and weight prototypes must have the same shape"

        if normalize:
            geo = torch.nn.functional.normalize(geo_prototypes, dim=1)
            weight = torch.nn.functional.normalize(weight_prototypes, dim=1)
        else:
            geo = geo_prototypes
            weight = weight_prototypes

        # [K, K] cross similarity
        sim_matrix = torch.matmul(geo, weight.t()).detach().cpu().numpy()
        sim_percent = sim_matrix * 100.0

        class_labels = [f"{idx_to_class[i][1]}" for i in range(len(idx_to_class))]

        n = len(class_labels)
        size = max(6, n * 0.5)

        plt.figure(figsize=(size, size))
        ax = sns.heatmap(
            sim_percent,
            cmap="viridis",
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot=True,
            fmt=".0f",
            vmin=-100.0,
            vmax=100.0,
            square=True,
            cbar_kws={"label": "Cosine Similarity (%)"},
        )

        # 统一 colorbar 为百分比语义
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-100, -50, 0, 50, 100])
        cbar.set_ticklabels(["-100%", "-50%", "0%", "50%", "100%"])

        plt.xlabel("Weight Prototypes")
        plt.ylabel("Geometric Prototypes")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, filename),
            bbox_inches="tight",
        )
        plt.close()
