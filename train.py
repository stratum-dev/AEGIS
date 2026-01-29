import os
import time
from utils.trainer import Trainer, save_training_config
from utils.seed import set_seed
from utils.dataset import VulnerabilityDataset
from transformers import RobertaTokenizer
from datasets import load_dataset
from utils.config import ModelConfig
from utils.logger import log

SUBSET_NAME = "megavul-mini"
DATASET_NAME = "codemetic/AEGIS"

# Fill your device here. "cpu","cuda:0","cuda:1", etc.
DEVICE = "cuda:1"

# Output directory
OUTPUT_MODEL_DIR = f"model_{SUBSET_NAME}_{time.strftime('%Y%m%d-%H-%M-%S')}"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

PROTOTYPE_HEATMAP_OUTPUT_DIR = os.path.join(
    OUTPUT_MODEL_DIR, "val_geo_prototype_similarity"
)
os.makedirs(PROTOTYPE_HEATMAP_OUTPUT_DIR, exist_ok=True)

UMAP_OUTPUT_DIR = os.path.join(OUTPUT_MODEL_DIR, "val_umap")
os.makedirs(UMAP_OUTPUT_DIR, exist_ok=True)

PROROTYPE_SIMILARITY_OUTPUT_DIR = os.path.join(
    OUTPUT_MODEL_DIR, "val_geo_weight_prototype_similarity"
)
os.makedirs(PROROTYPE_SIMILARITY_OUTPUT_DIR, exist_ok=True)

log.set_log_file(os.path.join(OUTPUT_MODEL_DIR, "train.log"))


def main():
    config = ModelConfig(
        subset_name=SUBSET_NAME,
        dataset_name=DATASET_NAME,
        model_name="microsoft/unixcoder-base",
        max_checkpoints=1,
        max_length=512,
        batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_epochs=1000,
        early_stopping_patience=100,
        random_seed=42,
        gamma=0.7,
        temperature=0.2,
        m0=0.8,
        s=30,
        momentum=0.99,
        output_dir=OUTPUT_MODEL_DIR,
        device=DEVICE,
        prototype_heatmap_output_dir=PROTOTYPE_HEATMAP_OUTPUT_DIR,
        umap_output_dir=UMAP_OUTPUT_DIR,
        prototype_similarity_output_dir=PROROTYPE_SIMILARITY_OUTPUT_DIR,
    )
    set_seed(config.RANDOM_SEED)

    log.print("🚀 Starting training from scratch...")
    log.print(f"Saved on position: {OUTPUT_MODEL_DIR}")

    save_training_config(config)

    # Load data
    dataset = load_dataset(config.DATASET_NAME, config.SUBSET_NAME)
    train_data, val_data, test_data = dataset["train"], dataset["val"], dataset["test"]

    tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = VulnerabilityDataset(train_data, tokenizer, config.MAX_LENGTH)
    val_dataset = VulnerabilityDataset(val_data, tokenizer, config.MAX_LENGTH)
    test_dataset = VulnerabilityDataset(test_data, tokenizer, config.MAX_LENGTH)

    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        config=config,
    )
    log.print(f"Total classes: {trainer.num_classes}")
    trainer.train()


if __name__ == "__main__":
    main()
