import math
from typing import Tuple
import torch
from transformers import RobertaConfig, RobertaModel
import torch.nn as nn
import torch.nn.functional as F
from utils.calc import l2_norm


class RoBERTaEncoder(nn.Module):
    def __init__(self, model_name: str, layers_to_concat: Tuple[int, ...]):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.layers_to_concat = layers_to_concat
        self.hidden_size = self.config.hidden_size
        self.feature_dim = self.hidden_size

        self.num_layers_to_use = len(layers_to_concat)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers_to_use))
        self.softmax = nn.Softmax(dim=0)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = (
            outputs.hidden_states
        )  # tuple of 13 tensors: [emb, L1, L2, ..., L12]

        selected_layers = []
        for layer_idx in self.layers_to_concat:
            selected_layers.append(hidden_states[layer_idx])  # (B, L, D)

        stacked = torch.stack(selected_layers, dim=0)  # (N, B, L, D)
        norm_weights = self.softmax(self.layer_weights)  # (N,)
        weighted = norm_weights.view(-1, 1, 1, 1) * stacked
        fused = torch.sum(weighted, dim=0)  # (B, L, D)
        cls_embedding = fused[:, 0, :]  # (B, D)
        return cls_embedding


class KappaLossClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, s, m0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m0 = m0
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    # def cos_m_theta(cos_theta, m):
    #     theta = torch.acos(cos_theta)
    #     k = (m * theta / math.pi).floor()
    #     phi_theta = ((-1.0) ** k) * torch.cos(m * theta) - 2 * k
    #     return phi_theta

    # def forward(self, x, labels, margins):
    #     """
    #     Pseudo-SphereFace with per-class margin (NOT standard SphereFace)
    #     """

    #     weight_norm = l2_norm(self.weight)
    #     cos_theta = torch.mm(x, weight_norm.t())
    #     cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    #     theta = torch.acos(cos_theta)
    #     one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

    #     # margins: (C,)
    #     m_gt = margins[labels]  # (B,)
    #     m_gt = m_gt.view(-1, 1)  # (B, 1)

    #     # Apply per-sample angular margin
    #     phi_theta = torch.cos(m_gt * theta)

    #     cos_theta_m = one_hot * phi_theta + (1.0 - one_hot) * cos_theta
    #     logits = self.s * cos_theta_m

    #     return logits

    # def forward(self, x, labels,margins):
    #     # Normalize input features and weight
    #     x_norm = l2_norm(x)
    #     weight_norm = l2_norm(self.weight)

    #     # Cosine similarity: (B, C)
    #     cos_theta = torch.mm(x_norm, weight_norm.t())
    #     # Clamp for numerical stability
    #     cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    #     # Convert to angle
    #     theta = torch.acos(cos_theta)

    #     # Create one-hot labels
    #     one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

    #     # Add angular margin only to the true class
    #     theta_plus_m = theta + one_hot * margins

    #     # Convert back to cosine
    #     cos_theta_plus_m = torch.cos(theta_plus_m)

    #     # Scale logits
    #     logits = self.s * cos_theta_plus_m
    #     return logits

    def forward(self, x, labels, margins):
        # Normalize weight
        weight_norm = l2_norm(self.weight)
        # Cosine similarity matrix: (B, C)
        cos_theta = torch.mm(x, weight_norm.t())
        # Clamp for numerical stability
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Add margins to true class
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        margins_expanded = margins.unsqueeze(0).expand_as(cos_theta)
        cos_theta_m = cos_theta - one_hot * margins_expanded

        # Scale
        logits = self.s * cos_theta_m
        return logits

    # def forward(self, x, labels, margins):
    #     weight_norm = l2_norm(self.weight)

    #     cos_theta = torch.mm(x, weight_norm.t())
    #     cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    #     margins_expanded = margins.unsqueeze(0).expand_as(cos_theta)

    #     cos_m = torch.cos(margins_expanded)
    #     sin_m = torch.sin(margins_expanded)

    #     # sqrt 仍然存在，但不会 acos，也不会反三角
    #     sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=1e-7))

    #     # cos(theta + m)
    #     phi = cos_theta * cos_m - sin_theta * sin_m

    #     # === 关键：单调性保护 ===
    #     th = torch.cos(math.pi - margins_expanded)
    #     mm = torch.sin(math.pi - margins_expanded) * margins_expanded

    #     phi = torch.where(cos_theta > th, phi, cos_theta - mm)

    #     one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
    #     logits = self.s * (one_hot * phi + (1.0 - one_hot) * cos_theta)

    #     return logits


class AEGISModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        s,
        m0,
    ):
        super().__init__()
        self.encoder = RoBERTaEncoder(model_name, (8, 9, 10, 11))
        self.feature_dim = self.encoder.feature_dim
        self.kappaface_head = KappaLossClassifierHead(
            self.encoder.feature_dim, num_classes, s, m0
        )

    def forward(self, input_ids, attention_mask, labels=None, margins=None):
        features = self.encoder(input_ids, attention_mask)
        features_norm = l2_norm(features)
        if labels is not None and margins is not None:
            logits = self.kappaface_head(features_norm, labels, margins)
            return features_norm, logits
        else:
            return features_norm
