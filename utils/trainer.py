from datetime import datetime
import os
import time
import warnings
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
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
from utils.loss import kappa_loss, prototype_alignment_loss
from utils.checkpoint import save_checkpoint_with_limit
from utils.calc import (
    angular_variance,
    between_class_angular_margin,
    estimate_vmf_concentration,
    evaluate_etf_proximity,
    geodesic_variance,
    geometric_median,
    l2_normalize,
    mean_resultant_length,
    pairwise_angular_dispersion,
    spherical_davies_bouldin,
    spherical_silhouette_score,
)
from utils.logger import log
from utils.serialize import save_to_json

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
        ).detach()  # ✅ 修复：添加 detach()
        self.current_kappas_norm = torch.full(
            (self.num_classes,),
            0,
            device=self.train_config.DEVICE,
        )
        self.current_psis = None
        self.current_gamma = 0

        self.model: AEGISModel = AEGISModel(
            self.model_config.BACKBONE_REPO,
            self.num_classes,
            self.model_config.S0,
            self.model_config.M0,
        ).to(self.train_config.DEVICE)

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
        self.train_avg_prototypes = torch.zeros(
            self.num_classes, self.model.feature_dim, device=self.train_config.DEVICE
        )
        self.train_geo_prototypes = torch.zeros(
            self.num_classes, self.model.feature_dim, device=self.train_config.DEVICE
        )
        self.train_weight_prototypes = F.normalize(
            self.model.kappaface_head.weight.detach(), dim=1
        )

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
        psis = []

        # ===== estimate class-wise vMF concentration =====
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                kappas.append(1e-6)
                continue
            class_embs = embeddings[mask]
            kappa = estimate_vmf_concentration(class_embs)
            kappas.append(kappa)

        kappas = torch.tensor(kappas, device=self.train_config.DEVICE).clamp(min=1e-6)

        C = self.num_classes
        s0 = self.model_config.S0

        # ============================================================
        # ✅ Scheme 1: log-kappa softmax relative scaling (stable)
        # ============================================================

        u = torch.log(kappas)
        r = torch.softmax(u / C, dim=0) * C  # mean = 1
        scales = s0 * r  # mean = s0

        # ============================================================

        # ===== difficulty-aware weight omega_k =====
        kappas_np = kappas.detach().cpu().numpy()
        mu_k = np.mean(kappas_np)
        sigma_k = np.std(kappas_np) + 1e-8
        kappas_norm = torch.from_numpy((kappas_np - mu_k) / sigma_k).to(
            self.train_config.DEVICE
        )

        omega_k = 1 - torch.sigmoid(0.5 * kappas_norm)  # [C]

        # ===== frequency-aware weight omega_n =====
        omega_f_list = []
        for c in range(self.num_classes):
            n_c = self.class_counts.get(c, 1)
            k = self.max_class_count
            w = (torch.cos(torch.pi * torch.tensor(n_c) / k) + 1) / 2
            omega_f_list.append(w.item())

        omega_f = torch.tensor(omega_f_list, device=self.train_config.DEVICE)

        # ===== Adaptive gamma via kappa stability =====
        # gamma reflects reliability of kappa estimation
        # larger variance -> rely more on concentration

        # kappa_std = torch.std(kappas)
        # kappa_mean = torch.mean(kappas).clamp(min=1e-6)

        # gamma = (kappa_std / (kappa_std + kappa_mean)).clamp(0.0, 1.0).item()
        gamma = 0.5
        # ===== final margin =====
        for c in range(self.num_classes):
            psi = (gamma) * omega_k[c].item() + (1 - gamma) * omega_f[c].item()
            psis.append(psi)
            margins[c] = self.model_config.M0 * psi

        return (margins.detach(), scales.detach(), kappas_norm.detach(), psis, gamma)

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
        embeddings = l2_normalize(np.array(embeddings))

        silhouette_avg = (
            spherical_silhouette_score(embeddings, pred_labels)
            if len(set(pred_labels)) > 1
            else 0
        )

        dbi_score = (
            spherical_davies_bouldin(embeddings, pred_labels)
            if len(set(pred_labels)) > 1
            else float("inf")
        )

        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        ami_score = adjusted_mutual_info_score(true_labels, pred_labels)
        ari_score = adjusted_rand_score(true_labels, pred_labels)

        return {
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

        # ✅ 新增：用于记录验证集损失
        total_val_loss = 0.0
        total_val_kappa_loss = 0.0
        total_val_reg_loss = 0.0
        num_val_batches = 0

        self.model.eval()

        # 获取当前的权重原型用于计算正则化损失 (与训练时保持一致)
        # 注意：这里使用 detach() 因为验证时不更新权重
        weight_prototypes = F.normalize(
            self.model.kappaface_head.weight.detach(), dim=1
        )

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating val at epoch {epoch}"):
                input_ids = batch["input_ids"].to(self.train_config.DEVICE)
                attention_mask = batch["attention_mask"].to(self.train_config.DEVICE)
                truth_class_keys = batch["class_key"]
                truth_class_indices = torch.tensor(
                    [self.class_to_index[k] for k in truth_class_keys],
                    device=self.train_config.DEVICE,
                )

                # ✅ 前向传播 (保持与训练时相同的参数传递)
                embs, logits = self.model(
                    input_ids,
                    attention_mask,
                    truth_class_indices,
                    scales=self.current_scales,
                    margins=self.current_margins,
                )

                # ✅ 计算验证集损失
                # 注意：验证时通常不需要 autocast，或者即使需要也不影响 no_grad 下的计算
                # 为了数值稳定性，我们直接计算 float32 下的 loss，或者如果显存紧张也可以包一层 autocast
                # 这里为了简单和准确，直接在 no_grad 下计算
                val_loss_kappa = kappa_loss(logits, truth_class_indices)
                val_loss_reg = prototype_alignment_loss(
                    embs, truth_class_indices, weight_prototypes
                )
                val_loss = val_loss_kappa + val_loss_reg

                # 累加损失 (item() 获取标量值)
                total_val_loss += val_loss.item()
                total_val_kappa_loss += val_loss_kappa.item()
                total_val_reg_loss += val_loss_reg.item()
                num_val_batches += 1

                all_val_embeddings.append(embs.detach().cpu().numpy())

                pred_class_indices = MetricCalculator.hierarchical_decision(
                    embs,
                    self.train_weight_prototypes,
                )
                all_pred_class_indices.extend(pred_class_indices)
                all_truth_labels.extend(batch["label"])
                all_truth_class_keys.extend(truth_class_keys)

        # ✅ 计算平均损失
        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_kappa_loss = total_val_kappa_loss / num_val_batches
            avg_val_reg_loss = total_val_reg_loss / num_val_batches
        else:
            avg_val_loss = avg_val_kappa_loss = avg_val_reg_loss = 0.0

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
        all_val_embeddings = np.concatenate(all_val_embeddings, axis=0)
        clustering_metrics = self._calculate_clustering_metrics(
            all_val_embeddings,
            [self.class_to_index[k] for k in all_truth_class_keys],
            all_pred_class_indices,
        )

        mrl = mean_resultant_length(all_val_embeddings)
        angular_var = angular_variance(all_val_embeddings)
        pariwise_dispersion = pairwise_angular_dispersion(all_val_embeddings)
        geodesic_var = geodesic_variance(all_val_embeddings)
        bcm = between_class_angular_margin(all_val_embeddings, all_pred_class_indices)

        etf_status = evaluate_etf_proximity(self.train_geo_prototypes)

        # 可视化部分保持不变
        VisualizationHelper.draw_plot_umap(
            all_val_embeddings,
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

        binary_file_path = os.path.join(
            self.train_config.BINARY_METRICS_OUTPUT_DIR, f"{epoch}.json"
        )
        cwe_file_path = os.path.join(
            self.train_config.CWE_METRICS_OUTPUT_DIR, f"{epoch}.json"
        )

        save_to_json(binary_metrics, binary_file_path)
        save_to_json(cwe_metrics, cwe_file_path)

        # ✅ 返回新增的损失值
        return (
            avg_val_loss,
            avg_val_kappa_loss,
            avg_val_reg_loss,
            mrl,
            angular_var,
            pariwise_dispersion,
            geodesic_var,
            bcm,
            clustering_metrics,
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
                generator=g,
                shuffle=True,
                collate_fn=custom_collate_fn,
            )
            log.print(
                f"\nEpoch {epoch}/{self.train_config.MAX_EPOCHES} - At {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            epoch_start_time = time.time()
            self.model.train()

            total_kappa_loss = 0.0
            total_reg_loss = 0.0
            total_combined_loss = 0.0

            # ===== 新增：用于累积特征的列表 =====
            online_embeddings = []
            online_truth_classes = []

            total_loss = 0.0
            num_samples_in_epoch = 0
            progress_bar = tqdm(train_loader, desc="Training")
            batch_start_time = time.time()

            # ===== 合并为单次遍历：同时提取特征和训练 =====
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
                        self.current_scales.detach(),
                        self.current_margins.detach(),
                    )
                    
                    # ✅【关键修复】立即 detach 并移至 CPU，防止显存泄漏
                    # 这样 GPU 显存会在下一个 batch 开始时被复用
                    online_embeddings.append(features_norm.detach().cpu())
                    online_truth_classes.append(truth_class_indices.detach().cpu())
                    
                    loss_kappa = kappa_loss(logits, truth_class_indices)

                    # ===== Prototype–Prototype Consistency =====
                    weight_prototypes = F.normalize(
                        self.model.kappaface_head.weight.detach(), dim=1
                    )
                    loss_reg = prototype_alignment_loss(
                        features_norm, truth_class_indices, weight_prototypes
                    )
                    loss = loss_kappa + loss_reg

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                total_reg_loss += loss_reg.item()
                total_kappa_loss += loss_kappa.item()
                total_combined_loss += loss.item()
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Kappa": f"{loss_kappa.item():.4f}",
                        "Reg": f"{loss_reg.item():.4f}",
                    }
                )

            # ===== 新增：epoch结束后统一计算自适应参数（供下一epoch使用）=====
            
            # ✅【关键修复】将 CPU 上的列表 concat，然后移回 GPU
            # 此时在线 embeddings 已经在 CPU 上了，concat 很快，然后再传到 GPU
            online_embeddings = torch.cat(online_embeddings, dim=0).to(self.train_config.DEVICE)
            online_truth_classes = torch.cat(
                online_truth_classes, dim=0
            ).to(self.train_config.DEVICE)

            # 基于在线特征计算自适应参数
            (
                self.current_margins,
                self.current_scales,
                self.current_kappas_norm,
                self.current_psis,
                self.current_gamma,
            ) = self._compute_adaptive_params(online_embeddings, online_truth_classes)
            
            # ... 后续代码保持不变 ...

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
            avg_train_reg_loss = total_reg_loss / num_batches

            # ===== 保存当前epoch的原型（从在线编码器计算）=====
            self.train_avg_prototypes = self._compute_average_prototypes(
                online_embeddings, online_truth_classes
            ).detach()
            self.train_geo_prototypes = self._compute_geometric_prototypes(
                online_embeddings, online_truth_classes
            ).detach()
            self.train_weight_prototypes = F.normalize(
                self.model.kappaface_head.weight.detach(), dim=1
            )

            (
                avg_combined_val_loss,
                avg_val_kappa_loss,
                avg_val_reg_loss,
                mrl,
                angular_var,
                pariwise_dispersion,
                geodesic_var,
                bcm,
                clustering_metrics,
                binary_metrics,
                cwe_metrics,
                etf_status,
            ) = self._evaluate_epoch(val_loader, epoch)

            log.print("Gamma: ", self.current_gamma)
            log.print("Scales: ", self.current_scales)
            log.print("Margins: ", self.current_margins)
            log.print("Kappas (Norm): ", self.current_kappas_norm)
            log.print("Psis: ", [f"{x:.4f}" for x in self.current_psis])

            log.print(
                f"Norm Score:{etf_status['norm_std']:.4f} | "
                f"ETF Mean Cos:{etf_status['mean_cos']:.4f} | "
                f"Std Cos:{etf_status['std_cos']:.4f} | "
                f"Frame Potential Ratio:{etf_status['fp_ratio']:.4f} | "
                f"ETF Score:{etf_status['etf_score']:.4f}"
            )

            log.print(
                f"[TRAIN LOSS] Combined: {avg_train_combined_loss:.4f} | "
                f"Kappa: {avg_train_kappa_loss:.4f} | "
                f"Reg: {avg_train_reg_loss:.4f}"
            )
            log.print(
                f"[VAL LOSS] Combined: {avg_combined_val_loss:.4f} | "
                f"Kappa: {avg_val_kappa_loss:.4f} | "
                f"Reg: {avg_val_reg_loss:.4f}"
            )

            log.print(
                f"SH: {clustering_metrics['silhouette_score']} | "
                f"NMI: {clustering_metrics['nmi_score']} | "
                f"AMI: {clustering_metrics['ami_score']} | "
                f"ARI: {clustering_metrics['ari_score']} | "
                f"DBI: {clustering_metrics['dbi_score']} | "
                f"MRL: {mrl} | "
                f"Angular Var: {angular_var} | "
                f"Pairwise Disperation (mean,std): {pariwise_dispersion} | "
                f"Geodesic Var: {geodesic_var} | "
                f"Between Class Margin (min,mean): {bcm} | "
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

            # ... (前文计算 metrics 代码不变)

            new_point = {
                "epoch": epoch,
                "binary_f1": binary_metrics["f1"],
                "macro_f1": cwe_metrics["macro"]["f1"],
            }
            
            # 1. 只检查是否被当前的 Pareto Front 支配
            # 不需要遍历 self.all_points (历史所有点)，只需遍历当前的最优解集
            is_dominated_by_front = False
            epsilon = 1e-6 # 防止浮点数噪声
            
            for p in self.pareto_front:
                if self._is_dominated(new_point, p):
                    is_dominated_by_front = True
                    break
            
            # 2. 如果未被当前前沿支配，则尝试加入
            if not is_dominated_by_front:
                # 在加入前，移除 front 中被新点支配的旧点
                # 注意：这里要创建一个新的列表，避免在遍历中修改列表
                new_pareto_front = []
                removed_count = 0
                
                for p in self.pareto_front:
                    # 如果旧点 p 被新点 new_point 支配，则丢弃 p
                    if self._dominates(new_point, p):
                        removed_count += 1
                    else:
                        # 检查是否重复（数值极其接近的点视为相同，避免无限微调导致的震荡）
                        if (abs(p["binary_f1"] - new_point["binary_f1"]) < epsilon and 
                            abs(p["macro_f1"] - new_point["macro_f1"]) < epsilon):
                            # 视为已存在，不添加新点，也不重置耐心值（或者视作无进步）
                            # 这里选择直接跳过，不视为新解
                            is_dominated_by_front = True # 标记为无需处理
                        else:
                            new_pareto_front.append(p)
                
                if not is_dominated_by_front:
                    new_pareto_front.append(new_point)
                    self.pareto_front = new_pareto_front
                    
                    # 只有当真正更新了 front (要么是纯新增，要么替换了旧点) 才算进步
                    # 这里的逻辑是：只要进入了 front，就算一次“发现”，重置早停
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
                        f"✅ New Pareto solution added! (Front size: {len(self.pareto_front)}) "
                        f"Binary F1: {binary_metrics['f1']:.4f}, Macro F1: {cwe_metrics['macro']['f1']:.4f}"
                    )
                    self.pareto_patience_counter = 0
                else:
                    # 这种情况是发现了一个与现有 front 点数值几乎一样的点
                    self.pareto_patience_counter += 1
                    log.print(f"⚠️ Duplicate solution found. Patience: {self.pareto_patience_counter}")
            else:
                self.pareto_patience_counter += 1
                log.print(
                    f"⚠️ Dominated by current front. Pareto patience: {self.pareto_patience_counter}/{self.train_config.EARLY_STOP_PATIENCE}"
                )

            log.print(
                f"⏱️ Epoch Training Time: {epoch_duration:.2f}s | "
                f"Average Training Time in Epoch: {avg_sample_time_ms:.2f}ms"
            )
            log.print(
                f"Epoch {epoch} Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            )

            if epoch >= self.train_config.MAX_EPOCHES - 1:
                log.print(
                    f"🛑 Stopping triggered after reaching max epoches: {self.train_config.MAX_EPOCHES}."
                )
                break

            if self.pareto_patience_counter >= self.train_config.EARLY_STOP_PATIENCE:
                log.print(
                    f"🛑 Pareto early stopping triggered after {self.train_config.EARLY_STOP_PATIENCE} epochs without new non-dominated solution."
                )
                break
