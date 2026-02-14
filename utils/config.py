import os


class ModelConfig:
    """模型配置定义（所有字段由外部传入）"""

    def __init__(
        self,
        subset_name: str,
        dataset_repo: str,
        backbone_repo: str,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        random_seed: int,
        s0: float,
        momentum: float,
    ):
        # 数据集
        self.SUBSET_NAME = subset_name
        self.DATASET_REPO = dataset_repo
        self.BACKBONE_REPO = backbone_repo

        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.WEIGHT_DECAY = weight_decay
        self.RANDOM_SEED = random_seed

        self.S0 = s0
        self.MOMENTUM = momentum


class TrainConfig:
    def __init__(
        self,
        device: str,
        output_dir: str,
        max_checkpoints: int,
        max_epoches: int,
        early_stop_patience: int,
    ):
        self.DEVICE = device
        self.OUTPUT_DIR = output_dir
        self.PROTOTYPE_ALIGNMENT_OUTPUT_DIR = os.path.join(
            self.OUTPUT_DIR, "val_prototype_alignment_matrix"
        )
        self.UMAP_OUTPUT_DIR = os.path.join(self.OUTPUT_DIR, "val_umap")
        self.PROTOTYPE_SIMILARITY_OUTPUT_DIR = os.path.join(
            self.OUTPUT_DIR, "val_prototype_similarity_matrix"
        )
        self.MAX_CHECKPOINTS = max_checkpoints
        self.MAX_EPOCHES = max_epoches
        self.EARLY_STOP_PATIENCE = early_stop_patience

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.PROTOTYPE_ALIGNMENT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.UMAP_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.PROTOTYPE_SIMILARITY_OUTPUT_DIR, exist_ok=True)


class EvalConfig:
    """模型配置类"""

    def __init__(
        self,
        model_dir: str,
        checkpoint: int,
        batch_size: int,
        device: str,
    ):
        self.MODEL_DIR = model_dir
        self.CHECKPOINT = checkpoint
        self.EVALUATION_OUTPUT_DIR = os.path.join(
            self.MODEL_DIR, "evaluations", f"evaluation-{checkpoint}"
        )
        self.BATCH_SIZE = batch_size

        self.EVAL_SPLIT = "test"

        self.DEVICE = device

        os.makedirs(self.EVALUATION_OUTPUT_DIR, exist_ok=True)
