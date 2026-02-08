import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer
from utils.config import EvalConfig, ModelConfig
from utils.aegis import AEGISModel
from utils.dataset import VulnerabilityDataset, custom_collate_fn
from utils.metrics import MetricCalculator
from utils.visual import VisualizationHelper


class Evaluator:
    def __init__(self, eval_config: EvalConfig):
        self.eval_config = eval_config
        self.model_config = None
        self.model = None
        self.geo_prototypes = None
        self.weight_prototypes = None
        self.idx_to_class = None
        self.test_loader = None
        self.testset_size = 0

    def load_model_and_checkpoint(self):
        checkpoint_path = os.path.join(
            self.eval_config.MODEL_DIR,
            f"pareto_checkpoint_epoch_{self.eval_config.CHECKPOINT}.pth",
        )
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint {self.eval_config.CHECKPOINT} is not found at {checkpoint_path}"

        print(f"Loading full checkpoint from: {checkpoint_path}")
        full_checkpoint = torch.load(
            checkpoint_path, map_location=self.eval_config.DEVICE, weights_only=False
        )

        class_to_idx = full_checkpoint["class_to_idx"]
        self.idx_to_class = full_checkpoint["idx_to_class"]
        model_config: ModelConfig = full_checkpoint["model_config"]
        self.model_config = model_config
        num_classes = len(class_to_idx)

        self.model = AEGISModel(
            model_config.BACKBONE_REPO,
            num_classes,
            model_config.M0,
            model_config.S,
        ).to(self.eval_config.DEVICE)
        self.model.load_state_dict(full_checkpoint["model_state_dict"], strict=True)
        self.model.eval()

        self.geo_prototypes = full_checkpoint["train_geo_prototypes"].to(
            self.eval_config.DEVICE
        )
        self.weight_prototypes = full_checkpoint["train_weight_prototypes"].to(
            self.eval_config.DEVICE
        )

    def prepare_data(self):
        """准备评估数据集"""
        tokenizer = RobertaTokenizer.from_pretrained(self.eval_config.BACKBONE_REPO)
        test_data = load_dataset(self.eval_config.DATASET_REPO, self.eval_config.SUBSET_NAME)[
            self.eval_config.EVAL_SPLIT
        ]
        test_dataset = VulnerabilityDataset(
            test_data, tokenizer, self.model_config.MAX_LENGTH
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.eval_config.BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.testset_size = len(test_data)
        print(f"Loaded {self.eval_config.EVAL_SPLIT} set with {self.testset_size} samples.")

    def run_evaluation(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, List[str]]:
        """执行评估并返回指标与嵌入"""
        all_pred_class_indices = []
        all_truth_labels = []
        all_truth_class_keys = []
        all_test_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                self.test_loader, desc=f"Evaluating {self.eval_config.EVAL_SPLIT}"
            ):
                embs = self.model(
                    batch["input_ids"].to(self.eval_config.DEVICE),
                    batch["attention_mask"].to(self.eval_config.DEVICE),
                )
                pred_indices = MetricCalculator.hierarchical_decision(
                    embs, self.geo_prototypes
                )
                all_pred_class_indices.extend(pred_indices)
                all_truth_labels.extend(batch["label"])
                all_truth_class_keys.extend(batch["class_key"])
                all_test_embeddings.append(embs.cpu().numpy())

        # Binary prediction
        y_true_binary = np.array(all_truth_labels)
        y_pred_binary = np.array(
            [self.idx_to_class[idx][0] for idx in all_pred_class_indices]
        )

        binary_metrics = MetricCalculator.calculate_l1_metrics(
            y_true_binary, y_pred_binary
        )
        cwe_metrics = MetricCalculator.calculate_l2_metrics(
            all_pred_class_indices, all_truth_class_keys, self.idx_to_class
        )

        all_test_embeddings_array = np.concatenate(all_test_embeddings, axis=0)
        return (
            binary_metrics,
            cwe_metrics,
            all_test_embeddings_array,
            all_truth_class_keys,
        )

    def save_results(self, binary_metrics: Dict, cwe_metrics: Dict):
        """保存评估指标到 JSON 文件"""
        os.makedirs(self.eval_config.EVALUATION_OUTPUT_DIR, exist_ok=True)

        # Binary metrics
        binary_file = os.path.join(
            self.eval_config.EVALUATION_OUTPUT_DIR, "binary-metrics.json"
        )
        for k, v in binary_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        with open(binary_file, "w") as f:
            json.dump(binary_metrics, f, indent=4)
        print(f"Binary results saved to: {binary_file}")

        # CWE metrics
        cwe_file = os.path.join(
            self.eval_config.EVALUATION_OUTPUT_DIR,
            f"multi-class-cwe-metrics-{self.eval_config.EVAL_SPLIT}.json",
        )
        with open(cwe_file, "w") as f:
            json.dump(cwe_metrics, f, indent=4)
        print(f"CWE results saved to: {cwe_file}")

        # 打印摘要
        macro_mcc = cwe_metrics["macro"]["mcc"]
        macro_f1 = cwe_metrics["macro"]["f1"]
        hier_acc = cwe_metrics["hier_acc"]
        print(
            f"\nSummary:\nMacro MCC: {macro_mcc:.4f}, Macro F1: {macro_f1:.4f}, Hier Acc: {hier_acc:.4f}"
        )

    def create_visualizations(self, embeddings: np.ndarray, true_class_keys: List[str]):
        """生成 UMAP 和原型热力图"""
        VisualizationHelper.draw_plot_umap(
            embeddings,
            true_class_keys,
            self.eval_config.EVALUATION_OUTPUT_DIR,
            "umap.svg",
            f"{self.eval_config.SUBSET_NAME} - Evaluation",
        )
        VisualizationHelper.draw_prototype_similarity_matrix(
            self.geo_prototypes,
            self.idx_to_class,
            self.eval_config.EVALUATION_OUTPUT_DIR,
            "prototype-similarity.svg",
            f"{self.eval_config.SUBSET_NAME} - Evaluation",
        )
        VisualizationHelper.draw_prototype_alignment_matrix(
            self.geo_prototypes,
            self.weight_prototypes,
            self.idx_to_class,
            self.eval_config.EVALUATION_OUTPUT_DIR,
            "prototype-alignment.svg",
            f"{self.eval_config.SUBSET_NAME} - Evaluation",
        )

    def evaluate(self):
        """主评估流程"""
        self.load_model_and_checkpoint()
        self.prepare_data()
        binary_metrics, cwe_metrics, embeddings, true_class_keys = self.run_evaluation()
        self.save_results(binary_metrics, cwe_metrics)
        self.create_visualizations(embeddings, true_class_keys)
