import os
from typing import Tuple
import torch


class ModelConfig:
    """模型配置定义（所有字段由外部传入）"""

    def __init__(
        self,
        # ===== 数据集配置 =====
        subset_name: str,
        dataset_name: str,
        model_name: str,
        # ===== 训练配置 =====
        max_checkpoints: int,
        # ===== 模型层配置 =====
        roberta_layers_to_concat: Tuple[int, ...],
        # ===== 超参数 =====
        max_length: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        early_stopping_patience: int,
        random_seed: int,
        # ===== Kappa / Prototype Loss =====
        gamma: float,
        temperature: float,
        m0: float,
        s: float,
        momentum: float,
        regular_ratio: float,
        # ===== 设备 =====
        device: torch.device,
        # ===== 输出目录 =====
        output_dir: str,
        prototype_heatmap_output_dir: str,
        umap_output_dir: str,
        prototype_similarity_output_dir: str,
    ):
        # 数据集
        self.SUBSET_NAME = subset_name
        self.DATASET_NAME = dataset_name
        self.MODEL_NAME = model_name

        # 训练
        self.MAX_CHECKPOINTS = max_checkpoints

        # 模型层
        self.ROBERTA_LAYERS_TO_CONCAT = roberta_layers_to_concat

        # 超参数
        self.MAX_LENGTH = max_length
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.WEIGHT_DECAY = weight_decay
        self.MAX_EPOCHS = max_epochs
        self.EARLY_STOPPING_PATIENCE = early_stopping_patience
        self.RANDOM_SEED = random_seed

        # Loss
        self.GAMMA = gamma
        self.TEMPERATURE = temperature
        self.M0 = m0
        self.S = s
        self.MOMENTUM = momentum
        self.REGULAR_RATIO = regular_ratio

        # 设备
        self.DEVICE = device

        # 输出目录
        self.OUTPUT_DIR = output_dir
        self.PROTOTYPE_HEATMAP_OUTPUT_DIR = prototype_heatmap_output_dir
        self.UMAP_OUTPUT_DIR = umap_output_dir
        self.PROTOTYPE_SIMILARITY_OUTPUT_DIR = prototype_similarity_output_dir


class EvalConfig:
    """模型配置类"""

    def __init__(
        self,
        model_dir: str,
        subset_name: str,
        model_name: str,
        checkpoint: int,
        dataset_name: str,
        batch_size: int,
        random_seed: int,
        device: str,
    ):
        # 模型路径和检查点配置
        self.MODEL_DIR = model_dir
        self.SUBSET_NAME = subset_name
        self.CHECKPOINT = checkpoint
        self.EVALUATION_OUTPUT_DIR = os.path.join(
            self.MODEL_DIR, f"evaluation-{checkpoint}"
        )

        # 数据集和模型配置
        self.DATASET_NAME = dataset_name
        self.MODEL_NAME = model_name
        self.MAX_LENGTH = 512
        self.BATCH_SIZE = batch_size

        # 可视化配置
        self.EVAL_SPLIT = "test"

        # 设备和随机种子
        self.DEVICE = device
        self.RANDOM_SEED = random_seed

        # 创建输出目录
        os.makedirs(self.EVALUATION_OUTPUT_DIR, exist_ok=True)
