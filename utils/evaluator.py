import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer

from utils.aegis import AEGISModel
from utils.dataset import VulnerabilityDataset, custom_collate_fn
from utils.metrics import MetricCalculator
from utils.visual import VisualizationHelper


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.geo_prototypes = None
        self.idx_to_class = None
        self.test_loader = None
        self.test_size = 0

    def load_model_and_checkpoint(self):
        """加载模型、原型和类映射"""
        model_path = os.path.join(
            self.config.MODEL_DIR,
            f"pareto_checkpoint_epoch_{self.config.CHECKPOINT}.pth",
        )
        assert os.path.exists(model_path), f"Model not found at {model_path}"

        print(f"Loading full checkpoint from: {model_path}")
        full_checkpoint = torch.load(
            model_path, map_location=self.config.DEVICE, weights_only=False
        )

        class_to_idx = full_checkpoint["class_to_idx"]
        self.idx_to_class = full_checkpoint["idx_to_class"]
        saved_config = full_checkpoint["config"]
        num_classes = len(class_to_idx)

        self.model = AEGISModel(
            saved_config.MODEL_NAME,
            num_classes,
            saved_config.M0,
            saved_config.S,
            saved_config.ROBERTA_LAYERS_TO_CONCAT,
        ).to(self.config.DEVICE)
        self.model.load_state_dict(full_checkpoint["model_state_dict"], strict=True)
        self.model.eval()

        self.geo_prototypes = full_checkpoint["train_geo_prototypes"].to(
            self.config.DEVICE
        )
        print(f"Loaded geo_prototypes of shape: {self.geo_prototypes.shape}")

    def prepare_data(self):
        """准备评估数据集"""
        tokenizer = RobertaTokenizer.from_pretrained(self.config.MODEL_NAME)
        test_data = load_dataset(self.config.DATASET_NAME, self.config.SUBSET_NAME)[
            self.config.EVAL_SPLIT
        ]
        test_dataset = VulnerabilityDataset(
            test_data, tokenizer, self.config.MAX_LENGTH
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.test_size = len(test_data)
        print(f"Loaded {self.config.EVAL_SPLIT} set with {self.test_size} samples.")

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
                self.test_loader, desc=f"Evaluating {self.config.EVAL_SPLIT}"
            ):
                embs = self.model(
                    batch["input_ids"].to(self.config.DEVICE),
                    batch["attention_mask"].to(self.config.DEVICE),
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
        os.makedirs(self.config.EVALUATION_OUTPUT_DIR, exist_ok=True)

        # Binary metrics
        binary_file = os.path.join(
            self.config.EVALUATION_OUTPUT_DIR, "binary-metrics.json"
        )
        for k, v in binary_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        with open(binary_file, "w") as f:
            json.dump(binary_metrics, f, indent=4)
        print(f"Binary results saved to: {binary_file}")

        # CWE metrics
        cwe_file = os.path.join(
            self.config.EVALUATION_OUTPUT_DIR,
            f"multi-class-cwe-metrics-{self.config.EVAL_SPLIT}.json",
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
            self.config.EVALUATION_OUTPUT_DIR,
            "umap.svg",
            f"{self.config.SUBSET_NAME} - Evaluation",
        )
        VisualizationHelper.draw_prototype_heatmap(
            self.geo_prototypes,
            self.idx_to_class,
            self.config.EVALUATION_OUTPUT_DIR,
            "prototype-heatmap.svg",
            f"{self.config.SUBSET_NAME} - Evaluation",
        )

    def evaluate(self):
        """主评估流程"""
        self.load_model_and_checkpoint()
        self.prepare_data()
        binary_metrics, cwe_metrics, embeddings, true_class_keys = self.run_evaluation()
        self.save_results(binary_metrics, cwe_metrics)
        self.create_visualizations(embeddings, true_class_keys)
