import os
from utils.trainer import Trainer
from utils.serialize import save_to_json
from utils.seed import set_seed
from utils.dataset import VulnerabilityDataset
from datasets import load_dataset
from utils.config import ModelConfig, TrainConfig
from utils.logger import log
from datetime import datetime

# ============================ Huggingface Repo =================================
# The dataset repository
DATASET_REPO = "codemetic/AEGIS-dataset"
# Subset for above repo.
# Avaliable at: "bigvul", "mvd", "megavul", "draper", "reposvul", "diversevul"
SUBSET_NAME = "draper"
# The backbone repository
# You can try these backbones also:
# "microsoft/graphcodebert-base", "microsoft/codebert-base", "microsoft/unixcoder-base"
# "Salesforce/codet5-base", "Salesforce/codet5p-220m", "Salesforce/codet5p-770m"
BACKBONE_REPO = "Salesforce/codet5-base"

# ============================ Hyperparameters ==================================
# The descriptions for these hyperparameters was intruduced in paper.
# Please refer the original paper to adjust the hyperparameters
S0 = 30
M0 = 0.3

BATCH_SIZE = 40
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# ============================ Training Settings=================================
# Fill your device here. "cuda","cuda:0","cuda:1","cuda:2", etc.
# Mixed-precision relies on CUDA, and therefore training on CPU is NOT supported.
DEVICE = "cuda:2"
MAX_EPOCHES = 100
EARLY_STOP_PATIENCE = 20
MAX_CHECKPOINTS = 0
OUTPUT_DIR = os.path.join(
    "models",
    f"aegis_{BACKBONE_REPO.split('/')[1]}_{SUBSET_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.set_log_file(os.path.join(OUTPUT_DIR, "train.log"))
    model_config = ModelConfig(
        subset_name=SUBSET_NAME,
        dataset_repo=DATASET_REPO,
        backbone_repo=BACKBONE_REPO,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        random_seed=RANDOM_SEED,
        s0=S0,
        m0=M0,
    )

    train_config = TrainConfig(
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        max_checkpoints=MAX_CHECKPOINTS,
        max_epoches=MAX_EPOCHES,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )

    save_to_json(model_config, os.path.join(OUTPUT_DIR, "model_config.json"))

    set_seed(RANDOM_SEED)

    log.print("🚀 Starting training from scratch...")
    log.print(f"Saved on position: {OUTPUT_DIR}")

    # Load data
    dataset = load_dataset(model_config.DATASET_REPO, model_config.SUBSET_NAME)
    train_data, val_data, test_data = dataset["train"], dataset["val"], dataset["test"]

    train_dataset = VulnerabilityDataset(train_data, model_config)
    val_dataset = VulnerabilityDataset(val_data, model_config)
    test_dataset = VulnerabilityDataset(test_data, model_config)

    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        train_config=train_config,
        model_config=model_config,
    )
    log.print(f"Total classes: {trainer.num_classes}")
    trainer.train()


if __name__ == "__main__":
    main()
