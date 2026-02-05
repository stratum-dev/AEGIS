# resume-train.py
import os
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer

from utils.seed import set_seed
from utils.dataset import VulnerabilityDataset
from utils.trainer import Trainer
from utils.config import ModelConfig

RESUME_OUTPUT_DIR = (
    "/home/MHFangGPU/AEGIS/result/model_aegis_megavul_20260201-16-52-44"  # 训练目录
)
RESUME_FROM_EPOCH = 21  # pareto_checkpoint_epoch_{EPOCH}.pth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # -----------------------------
    # 1. 加载 checkpoint
    # -----------------------------
    ckpt_path = os.path.join(
        RESUME_OUTPUT_DIR,
        f"pareto_checkpoint_epoch_{RESUME_FROM_EPOCH}.pth",
    )
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # -----------------------------
    # 2. 重建 config
    # -----------------------------
    config: ModelConfig = checkpoint["config"]
    config.OUTPUT_DIR = RESUME_OUTPUT_DIR
    config.DEVICE = DEVICE

    set_seed(config.RANDOM_SEED)

    print(f"🔁 Resuming training from epoch {checkpoint['epoch']}")

    # -----------------------------
    # 3. 加载数据
    # -----------------------------
    dataset = load_dataset(config.DATASET_NAME, config.SUBSET_NAME)
    train_data, val_data, test_data = (
        dataset["train"],
        dataset["val"],
        dataset["test"],
    )

    tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = VulnerabilityDataset(train_data, tokenizer, config.MAX_LENGTH)
    val_dataset = VulnerabilityDataset(val_data, tokenizer, config.MAX_LENGTH)
    test_dataset = VulnerabilityDataset(test_data, tokenizer, config.MAX_LENGTH)

    # -----------------------------
    # 4. 构建 Trainer
    # -----------------------------
    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        config=config,
    )

    # -----------------------------
    # 5. 恢复训练状态
    # -----------------------------
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.momentum_model.load_state_dict(checkpoint["model_state_dict"])

    trainer.class_to_index = checkpoint["class_to_idx"]
    trainer.index_to_class = checkpoint["idx_to_class"]

    trainer.train_avg_prototypes = checkpoint["train_avg_prototypes"].to(DEVICE)
    trainer.train_geo_prototypes = checkpoint["train_geo_prototypes"].to(DEVICE)

    trainer.start_epoch = checkpoint["epoch"] + 1

    print(f"✅ Model & prototypes loaded")
    print(f"➡️ Start epoch set to {trainer.start_epoch}")

    # -----------------------------
    # 6. 继续训练
    # -----------------------------
    trainer.train()


if __name__ == "__main__":
    main()
