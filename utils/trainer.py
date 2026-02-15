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
from utils.loss import kappa_loss, proto_loss_etf
from utils.checkpoint import save_checkpoint_with_limit
from utils.calc import (
    compute_class_separation_index,
    compute_dunn_index,
    estimate_vmf_concentration,
    evaluate_etf_proximity,
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
        self.maximum_margin = self.num_classes / (self.num_classes - 1)

        self.all_points = []
        self.pareto_front = []
        self.pareto_patience_counter = 0

        self.current_margins = torch.full(
            (self.num_classes,),
            self.maximum_margin,
            device=self.train_config.DEVICE,
        ).detach()
        self.current_scales = torch.full(
            (self.num_classes,),
            self.model_config.S0,
            device=self.train_config.DEVICE,
        )
        self.current_kappas = torch.full(
            (self.num_classes,),
            0,
            device=self.train_config.DEVICE,
        )

        self.model: AEGISModel = AEGISModel(
            self.model_config.BACKBONE_REPO,
            self.num_classes,
            self.model_config.S0,
            self.maximum_margin,
        ).to(self.train_config.DEVICE)

        self.momentum_model: AEGISModel = AEGISModel(
            self.model_config.BACKBONE_REPO,
            self.num_classes,
            self.model_config.S0,
            self.maximum_margin,
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
        device = self.train_config.DEVICE
        C = self.num_classes
        d = embeddings.shape[1]  # embedding dimension

        margins = torch.zeros(C, device=device)
        kappas = []

        # ===== estimate class-wise vMF concentration =====
        for c in range(C):
            mask = labels == c
            if mask.sum() == 0:
                kappas.append(1e-6)
                continue
            class_embs = embeddings[mask]
            kappa = estimate_vmf_concentration(class_embs)
            kappas.append(kappa)

        kappas = torch.tensor(kappas, device=device).clamp(min=1e-6)

        margins = torch.sqrt((d - 1) / kappas.detach())

        # ===== robust z-score normalization =====
        scales = torch.full(
            (self.num_classes,),
            self.model_config.S0,
            device=self.train_config.DEVICE,
        )

        return margins.detach(), scales.detach(), kappas.detach()

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
        etf_status = evaluate_etf_proximity(self.train_geo_prototypes)

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
            etf_status,
        )

    def train(self):

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        for epoch in range(self.start_epoch, self.train_config.MAX_EPOCHES):
            g = torch.Generator()
            g.manual_seed(self.model_config.RANDOM_SEED)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.model_config.BATCH_SIZE,
                shuffle=True,
                collate_fn=custom_collate_fn,
                generator=g,
            )

            log.print(
                f"\nEpoch {epoch}/{self.train_config.MAX_EPOCHES} - At {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            epoch_start_time = time.time()
            self.model.train()

            total_kappa_loss = 0.0
            total_combined_loss = 0.0

            # ===== 直接使用当前模型提取特征来构建原型 =====
            all_embeddings = []
            all_truth_classes = []

            self.model.eval()  # 切换到 eval 模式以冻结 BN/ Dropout
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Extracting features"):
                    embs = self.model.encoder(
                        batch["input_ids"].to(self.train_config.DEVICE),
                        batch["attention_mask"].to(self.train_config.DEVICE),
                    )
                    embs_norm = l2_norm(embs)
                    all_embeddings.append(embs_norm)
                    truth_class_indices = [
                        self.class_to_index[k] for k in batch["class_key"]
                    ]
                    all_truth_classes.extend(truth_class_indices)
            self.model.train()  # 切回 train 模式

            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_truth_classes = torch.tensor(
                all_truth_classes, device=self.train_config.DEVICE
            )

            global_avg_prototypes = self._compute_average_prototypes(
                all_embeddings, all_truth_classes
            )
            global_geo_prototypes = self._compute_geometric_prototypes(
                all_embeddings, all_truth_classes
            )
            (
                self.current_margins,
                self.current_scales,
                self.current_kappas,
            ) = self._compute_adaptive_params(all_embeddings, all_truth_classes)

            # ===== 正式训练阶段 =====
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
                    weight_prototypes = F.normalize(
                        self.model.kappaface_head.weight, dim=1
                    )
                    proto_loss = proto_loss_etf(
                        weight_prototypes
                    )

                    loss = loss_kappa + proto_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

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
                etf_status,
            ) = self._evaluate_epoch(val_loader, epoch)

            layer_weights = (
                torch.softmax(self.model.encoder.layer_weights, dim=0)
                .detach()
                .cpu()
                .numpy()
            )

            log.print("Margins: ", self.current_margins)
            log.print("Kappas: ", self.current_kappas)

            log.print(
                f"Combined Loss: {avg_train_combined_loss:.4f} | "
                f"Kappa Loss: {avg_train_kappa_loss:.4f}| "
                f"Proto Loss: {proto_loss.item():.4f}",
            )

            log.print(
                f"Norm Score:{etf_status['norm_std']:.3f} | "
                f"ETF Mean Cos:{etf_status['mean_cos']:.4f} | "
                f"Std Cos:{etf_status['std_cos']:.4f} | "
                f"Frame Potential Ratio:{etf_status['fp_ratio']:.3f} | "
                f"ETF Score:{etf_status['etf_score']:.3f}"
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
            log.print(f"Layer fusion weights: {dict(zip((8,9,10,11), layer_weights))}")

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
