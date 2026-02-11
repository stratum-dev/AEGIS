from datetime import datetime
import time
import warnings
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    silhouette_score,
)
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.config import ModelConfig, TrainConfig
from utils.metrics import MetricCalculator
from utils.dataset import custom_collate_fn
from utils.visual import VisualizationHelper
from utils.aegis import AEGISModel
from utils.loss import prototype_consistency_loss, kappa_loss
from utils.checkpoint import save_checkpoint_with_limit
from utils.calc import (
    compute_class_separation_index,
    compute_dunn_index,
    estimate_vmf_concentration,
    geometric_median,
    l2_norm,
)
from utils.logger import log

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        train_config: TrainConfig,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.train_config = train_config

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        class_counter = Counter(
            self.test_dataset[i]["class_key"] for i in range(len(self.test_dataset))
        )
        class_list = sorted(class_counter.keys(), key=lambda k: (-class_counter[k], k))

        self.class_to_index = {cls: idx for idx, cls in enumerate(class_list)}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
        self.num_classes = len(self.class_to_index)

        self.all_points = []
        self.pareto_front = []
        self.pareto_patience_counter = 0

        self.current_margins = torch.full(
            (self.num_classes,),
            self.model_config.M0,
            device=self.train_config.DEVICE,
        ).detach()
        self.current_scales = torch.full(
            (self.num_classes,),
            self.model_config.S0,
            device=self.train_config.DEVICE,
        )

        self.model: AEGISModel = AEGISModel(
            self.model_config.BACKBONE_REPO,
            self.num_classes,
            self.model_config.S0,
            self.model_config.M0,
        ).to(self.train_config.DEVICE)

        self.momentum_model: AEGISModel = AEGISModel(
            self.model_config.BACKBONE_REPO,
            self.num_classes,
            self.model_config.S0,
            self.model_config.M0,
        ).to(self.train_config.DEVICE)
        self.momentum_model.load_state_dict(self.model.state_dict())
        self.momentum_model.eval()
        for param in self.momentum_model.parameters():
            param.requires_grad = False

        self.log_lambda_ppc = torch.nn.Parameter(
            torch.zeros(1, device=self.train_config.DEVICE)
        )

        optimizer_params = [
            {"params": self.model.parameters()},
            {"params": [self.log_lambda_ppc], "lr": self.model_config.LEARNING_RATE},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.model_config.LEARNING_RATE,
            weight_decay=self.model_config.WEIGHT_DECAY,
        )
        self.scaler = GradScaler("cuda")
        self.global_step = 0

        self.class_counts = self._count_classes(train_dataset)
        self.max_class_count = max(self.class_counts.values())
        self.start_epoch = 0
        self.train_avg_prototypes = None
        self.train_geo_prototypes = None
        self.train_weight_prototypes = None

    def _is_dominated(self, new_point, existing_point):
        b_new, m_new = new_point["binary_f1"], new_point["macro_f1"]
        b_exist, m_exist = existing_point["binary_f1"], existing_point["macro_f1"]
        return (
            (b_exist >= b_new)
            and (m_exist >= m_new)
            and ((b_exist > b_new) or (m_exist > m_new))
        )

    def _dominates(self, point_a, point_b):
        return self._is_dominated(point_b, point_a)

    def _count_classes(self, dataset):
        counts = Counter(item["class_key"] for item in dataset)
        return {
            self.class_to_index[k]: v
            for k, v in counts.items()
            if k in self.class_to_index
        }

    def _compute_adaptive_params(self, embeddings, labels):
        margins = torch.zeros(self.num_classes, device=self.train_config.DEVICE)
        kappas = []

        # ===== estimate class-wise vMF concentration =====
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                kappas.append(1e-6)
                continue
            class_embs = embeddings[mask]
            kappa = estimate_vmf_concentration(class_embs)
            kappas.append(kappa)

        kappas = torch.tensor(kappas, device=self.train_config.DEVICE)

        # ===== class-wise adaptive scale =====
        scales = self.model_config.S0 * kappas / kappas.mean().clamp(min=1e-6)

        # ===== difficulty-aware weight omega_k =====
        kappas_np = kappas.detach().cpu().numpy()
        mu_k = np.mean(kappas_np)
        sigma_k = np.std(kappas_np) + 1e-8
        kappas_norm = torch.from_numpy((kappas_np - mu_k) / sigma_k).to(
            self.train_config.DEVICE
        )

        omega_k = torch.sigmoid(0.5 * kappas_norm)  # [C]

        # ===== frequency-aware weight omega_n =====
        omega_f_list = []
        for c in range(self.num_classes):
            n_c = self.class_counts.get(c, 1)
            k = self.max_class_count  # max |D_y|
            w = (torch.cos(torch.pi * torch.tensor(n_c) / k) + 1) / 2
            omega_f_list.append(w.item())

        omega_f_vec = torch.tensor(omega_f_list, device=self.train_config.DEVICE)

        # ===== Convert to probability distributions (sum to 1) =====
        def to_prob(x):
            return x / (x.sum() + 1e-8)

        p_k = to_prob(omega_k)
        p_n = to_prob(omega_f_vec)

        # ===== Adaptive gamma via Jensen-Shannon Divergence (JSD) =====
        # Compute midpoint distribution M = 0.5 * (P + Q)
        m = 0.5 * (p_k + p_n)

        # Compute KL(P || M) and KL(Q || M) with numerical stability
        eps = 1e-8
        kl_pm = torch.sum(p_k * torch.log((p_k + eps) / (m + eps)))
        kl_qm = torch.sum(p_n * torch.log((p_n + eps) / (m + eps)))

        jsd = 0.5 * (kl_pm + kl_qm)  # JSD ∈ [0, ln(2)] ≈ [0, 0.693]

        # Normalize JSD to [0, 1] for use as gamma (optional but safe)
        gamma = (jsd / np.log(2)).clamp(0.0, 1.0).item()

        # Optional: print for debugging
        log.print(f"JSD-based gamma: {gamma:.4f}")

        # ===== final margin =====
        for c in range(self.num_classes):
            omega_k_val = omega_k[c].item()
            omega_n_val = omega_f_vec[c].item()
            psi = gamma * omega_k_val + (1.0 - gamma) * omega_n_val
            m_c = self.model_config.M0 * (psi)
            margins[c] = m_c

        return margins, scales.detach()

    def _compute_average_prototypes(
        self, embeddings: torch.Tensor, class_indices: torch.Tensor
    ):
        num_classes = self.num_classes
        feat_dim = embeddings.size(1)
        prototypes = torch.zeros(num_classes, feat_dim, device=embeddings.device)
        prototypes.scatter_add_(
            0, class_indices.unsqueeze(1).expand(-1, feat_dim), embeddings
        )
        counts = torch.bincount(class_indices, minlength=num_classes).clamp(min=1)
        prototypes = prototypes / counts.unsqueeze(1)
        return F.normalize(prototypes, p=2, dim=1)

    def _compute_geometric_prototypes(
        self,
        embeddings: torch.Tensor,
        class_indices: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = torch.zeros(
            self.num_classes, embeddings.size(1), device=embeddings.device
        )
        class_iter = tqdm(
            range(self.num_classes),
            desc="Computing geometric prototypes",
            leave=False,
        )
        for c in class_iter:
            mask = class_indices == c
            if mask.any():
                prototypes[c] = geometric_median(embeddings[mask])
        return F.normalize(prototypes, p=2, dim=1)

    def _calculate_clustering_metrics(self, embeddings, true_labels, pred_labels):
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        embeddings = np.array(embeddings)

        ch_score = (
            calinski_harabasz_score(embeddings, pred_labels)
            if len(set(pred_labels)) > 1
            else 0.0
        )
        silhouette_avg = (
            silhouette_score(embeddings, pred_labels)
            if (len(set(pred_labels)) > 1 and len(set(pred_labels)) < len(embeddings))
            else 0
        )
        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        ami_score = adjusted_mutual_info_score(true_labels, pred_labels)
        ari_score = adjusted_rand_score(true_labels, pred_labels)
        dbi_score = (
            davies_bouldin_score(embeddings, pred_labels)
            if len(set(pred_labels)) > 1
            else float("inf")
        )

        return {
            "ch_score": ch_score,
            "silhouette_score": silhouette_avg,
            "nmi_score": nmi_score,
            "ami_score": ami_score,
            "ari_score": ari_score,
            "dbi_score": dbi_score,
        }

    def _evaluate_epoch(self, val_loader, epoch):
        all_pred_class_indices = []
        all_truth_labels = []
        all_truth_class_keys = []
        all_val_embeddings = []
        num_val_batches = 0

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating val at epoch {epoch}"):
                input_ids = batch["input_ids"].to(self.train_config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.train_config.DEVICE)
                truth_class_keys = batch["class_key"]
                truth_class_indices = torch.tensor(
                    [self.class_to_index[k] for k in truth_class_keys],
                    device=self.train_config.DEVICE,
                )

                embs, logits = self.model(
                    input_ids,
                    attention_mask,
                    truth_class_indices,
                    scales=self.current_scales,
                    margins=self.current_margins,
                )
                all_val_embeddings.append(embs.cpu().numpy())
                num_val_batches += 1

                pred_class_indices = MetricCalculator.hierarchical_decision(
                    embs,
                    self.train_geo_prototypes,
                )
                all_pred_class_indices.extend(pred_class_indices)
                all_truth_labels.extend(batch["label"])
                all_truth_class_keys.extend(truth_class_keys)

        truth_binary = np.array(all_truth_labels)
        pred_binary = np.array(
            [self.index_to_class[idx][0] for idx in all_pred_class_indices]
        )

        binary_metrics = MetricCalculator.calculate_l1_metrics(
            truth_binary, pred_binary
        )
        cwe_metrics = MetricCalculator.calculate_l2_metrics(
            all_pred_class_indices, all_truth_class_keys, self.index_to_class
        )

        clustering_metrics = self._calculate_clustering_metrics(
            np.concatenate(all_val_embeddings, axis=0),
            [self.class_to_index[k] for k in all_truth_class_keys],
            all_pred_class_indices,
        )

        val_embeddings_array = np.concatenate(all_val_embeddings, axis=0)
        dunn_score = compute_dunn_index(
            val_embeddings_array,
            np.array(all_pred_class_indices),
        )
        separation_score = compute_class_separation_index(
            val_embeddings_array,
            np.array([self.class_to_index[k] for k in all_truth_class_keys]),
        )

        VisualizationHelper.draw_plot_umap(
            val_embeddings_array,
            all_truth_class_keys,
            self.train_config.UMAP_OUTPUT_DIR,
            f"{epoch}.svg",
            f"{self.model_config.SUBSET_NAME} - Val Epoch - {epoch}",
        )
        VisualizationHelper.draw_prototype_similarity_matrix(
            self.train_geo_prototypes,
            self.index_to_class,
            self.train_config.PROTOTYPE_SIMILARITY_OUTPUT_DIR,
            f"{epoch}.svg",
            f"{self.model_config.SUBSET_NAME} - Val Epoch - {epoch}",
        )
        VisualizationHelper.draw_prototype_alignment_matrix(
            self.train_geo_prototypes,
            self.train_weight_prototypes,
            self.index_to_class,
            self.train_config.PROTOTYPE_ALIGNMENT_OUTPUT_DIR,
            f"{epoch}.svg",
            f"{self.model_config.SUBSET_NAME} - Val Epoch - {epoch}",
        )

        return (
            clustering_metrics["ch_score"],
            clustering_metrics["nmi_score"],
            clustering_metrics["ami_score"],
            clustering_metrics["ari_score"],
            clustering_metrics["dbi_score"],
            dunn_score,
            separation_score,
            binary_metrics,
            cwe_metrics,
        )

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.model_config.BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        for epoch in range(self.start_epoch, self.train_config.MAX_EPOCHES):
            log.print(
                f"\nEpoch {epoch}/{self.train_config.MAX_EPOCHES} - At {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            epoch_start_time = time.time()
            self.model.train()

            total_kappa_loss = 0.0
            total_combined_loss = 0.0

            momentum_embeddings = []
            momentum_truth_classes = []
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Updating momentum features"):
                    embs = self.momentum_model.encoder(
                        batch["input_ids"].to(self.train_config.DEVICE),
                        batch["attention_mask"].to(self.train_config.DEVICE),
                    )
                    embs_norm = l2_norm(embs)
                    momentum_embeddings.append(embs_norm)
                    truth_class_indices = [
                        self.class_to_index[k] for k in batch["class_key"]
                    ]
                    momentum_truth_classes.extend(truth_class_indices)
            momentum_embeddings = torch.cat(momentum_embeddings, dim=0)
            momentum_truth_classes = torch.tensor(
                momentum_truth_classes, device=self.train_config.DEVICE
            )
            self.current_margins, self.current_scales = self._compute_adaptive_params(
                momentum_embeddings, momentum_truth_classes
            )

            global_avg_prototypes = self._compute_average_prototypes(
                momentum_embeddings, momentum_truth_classes
            )
            global_geo_prototypes = self._compute_geometric_prototypes(
                momentum_embeddings, momentum_truth_classes
            )

            total_loss = 0.0
            num_samples_in_epoch = 0
            progress_bar = tqdm(train_loader, desc="Training")
            batch_start_time = time.time()
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.train_config.DEVICE)
                batch_size = len(batch["input_ids"])
                attention_mask = batch["attention_mask"].to(self.train_config.DEVICE)
                truth_class_indices = torch.tensor(
                    [self.class_to_index[k] for k in batch["class_key"]],
                    device=self.train_config.DEVICE,
                )

                num_samples_in_epoch += batch_size
                self.optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    features_norm, logits = self.model(
                        input_ids,
                        attention_mask,
                        truth_class_indices,
                        self.current_scales,
                        self.current_margins,
                    )
                    loss_kappa = kappa_loss(logits, truth_class_indices)
                    # ===== Prototype–Prototype Consistency =====
                    lambda_ppc = torch.exp(self.log_lambda_ppc.float())
                    avg_prototypes = global_avg_prototypes
                    weight_prototypes = F.normalize(
                        self.model.kappaface_head.weight.detach(), dim=1
                    )
                    loss_ppc = prototype_consistency_loss(
                        avg_prototypes,
                        weight_prototypes,
                    )

                    loss = loss_kappa + lambda_ppc * loss_ppc

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    for param_q, param_k in zip(
                        self.model.parameters(), self.momentum_model.parameters()
                    ):
                        param_k.data = (
                            self.model_config.MOMENTUM * param_k.data
                            + (1 - self.model_config.MOMENTUM) * param_q.data
                        )

                total_loss += loss.item()
                total_kappa_loss += loss_kappa.item()
                total_combined_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            train_duration = time.time() - batch_start_time
            avg_sample_time_ms = (
                (train_duration / num_samples_in_epoch) * 1000
                if num_samples_in_epoch > 0
                else 0
            )
            epoch_duration = time.time() - epoch_start_time
            num_batches = len(train_loader)
            avg_train_kappa_loss = total_kappa_loss / num_batches
            avg_train_combined_loss = total_combined_loss / num_batches

            (
                self.train_avg_prototypes,
                self.train_geo_prototypes,
                self.train_weight_prototypes,
            ) = (
                global_avg_prototypes,
                global_geo_prototypes,
                weight_prototypes,
            )
            (
                ch_score,
                nmi_score,
                ami_score,
                ari_score,
                dbi_score,
                dunn_score,
                separation_score,
                binary_metrics,
                cwe_metrics,
            ) = self._evaluate_epoch(val_loader, epoch)

            weights = (
                torch.softmax(self.model.encoder.layer_weights, dim=0)
                .detach()
                .cpu()
                .numpy()
            )

            log.print(self.current_scales)

            log.print(
                f"ppc/loss: {loss_ppc.item()}",
                f"ppc/lambda: {lambda_ppc.item()}",
                f"ppc/log_lambda: {self.log_lambda_ppc.item()}",
            )

            log.print(
                f"Train: Combined Loss: {avg_train_combined_loss:.4f} | "
                f"Kappa Loss: {avg_train_kappa_loss:.4f}"
            )

            log.print(
                f"CH Score: {ch_score:.4f} | "
                f"NMI: {nmi_score:.4f} | "
                f"AMI: {ami_score:.4f} | "
                f"ARI: {ari_score:.4f} | "
                f"DBI: {dbi_score:.4f} | "
                f"Dunn: {dunn_score:.4f} | "
                f"Separation: {separation_score:.4f}"
            )

            log.print(
                f"Val Confusion: "
                f"TP: {binary_metrics['tp']} | "
                f"TN: {binary_metrics['tn']} | "
                f"FP: {binary_metrics['fp']} | "
                f"FN: {binary_metrics['fn']}"
            )

            log.print(
                f"Vul/Non-vul Binary:  "
                f"MCC: {binary_metrics['mcc']:.4f} | "
                f"Binary F1: {binary_metrics['f1']:.4f} | "
                f"Binary Recall: {binary_metrics['recall']:.4f} | "
                f"Binary Precision: {binary_metrics['precision']:.4f}"
            )

            log.print(
                f"Multi CWE in Vul: "
                f"Macro MCC: {cwe_metrics['macro']['mcc']:.4f} | "
                f"Macro F1: {cwe_metrics['macro']['f1']:.4f} | "
                f"Macro Recall: {cwe_metrics['macro']['recall']:.4f} | "
                f"Macro Precision: {cwe_metrics['macro']['precision']:.4f} | "
                f"HierAcc: {cwe_metrics['hier_acc']:.4f}"
            )
            log.print(f"Layer fusion weights: {dict(zip((8,9,10,11), weights))}")

            new_point = {
                "epoch": epoch,
                "binary_f1": binary_metrics["f1"],
                "macro_f1": cwe_metrics["macro"]["f1"],
            }
            self.all_points.append(new_point)

            dominated = False
            for p in self.all_points:
                if p is not new_point and self._is_dominated(new_point, p):
                    dominated = True
                    break

            if not dominated:
                self.pareto_front = [
                    p for p in self.pareto_front if not self._dominates(new_point, p)
                ]
                self.pareto_front.append(new_point)
                save_checkpoint_with_limit(
                    model=self.model,
                    avg_proto=self.train_avg_prototypes,
                    geo_proto=self.train_geo_prototypes,
                    weight_proto=self.train_weight_prototypes,
                    class_to_idx=self.class_to_index,
                    model_config=self.model_config,
                    idx_to_class=self.index_to_class,
                    epoch=epoch,
                    output_dir=self.train_config.OUTPUT_DIR,
                    max_checkpoints=self.train_config.MAX_CHECKPOINTS,
                )
                log.print(
                    f"✅ New Pareto solution! Binary F1: {binary_metrics['f1']}, Macro F1: {cwe_metrics['macro']['f1']}"
                )
                self.pareto_patience_counter = 0
            else:
                self.pareto_patience_counter += 1
                log.print(
                    f"⚠️ Dominated solution. Pareto patience: {self.pareto_patience_counter}/{self.train_config.EARLY_STOP_PATIENCE}"
                )
            log.print(
                f"⏱️ Epoch Training Time: {epoch_duration:.2f}s | "
                f"Average Training Time in Epoch: {avg_sample_time_ms:.2f}ms"
            )
            log.print(
                f"Epoch {epoch} Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            )

            if epoch >= self.train_config.MAX_EPOCHES:
                log.print(
                    f"🛑 Stopping triggered after reaching max epoches: {self.train_config.MAX_EPOCHES}."
                )
                break

            if self.pareto_patience_counter >= self.train_config.EARLY_STOP_PATIENCE:
                log.print(
                    f"🛑 Pareto early stopping triggered after {self.train_config.EARLY_STOP_PATIENCE} epochs without new non-dominated solution."
                )
                break
