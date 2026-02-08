import os
from utils.trainer import Trainer, save_training_config
from utils.seed import set_seed
from utils.dataset import VulnerabilityDataset
from datasets import load_dataset
from utils.config import ModelConfig
from utils.logger import log

# Subset name
# "bigvul", "mvd", "megavul", "draper", "vuldeepecker", "reposvul"
SUBSET_NAME = "reposvul"

# Fill your device here. "cpu","cuda:0","cuda:1", etc.
DEVICE = "cuda:2"

# -------------- Huggingface Repo ---------------------------------
# The dataset repository
DATASET_NAME = "codemetic/AEGIS"

# The RoBERTa or T5 based backbone repository
# You can try these backbones also:
# "microsoft/graphcodebert-base", "microsoft/codebert-base", "microsoft/unixcoder-base"
# "Salesforce/codet5-base", "Salesforce/codet5p-220m", "Salesforce/codet5p-770m"
MODEL_NAME = "Salesforce/codet5p-220m"

# -------------- HyperParameters ----------------------------------
# The descriptions for these hyperparameters was intruduced in paper.
# Please refer the original paper to adjust the hyperparameters
GAMMA = 0.7
M0 = 0.8
S = 20
MOMENTUM = 0.99
TEMPERATURE = 0.2

# -------------- Training Settings---------------------------------
RANDOM_SEED = 42
BATCH_SIZE = 40
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
MAX_EPOCHES = 1000
EARLY_STOP_PATIENCE = 20

# The max length for processed code sample,
# depends on your Roberta backbone. 512 is default.
# Normally you don't need to modify this parameter.
MAX_LENGTHS = 512

# -------------- Output directory ---------------------------------
OUTPUT_MODEL_DIR = f"model_aegis_{SUBSET_NAME}"
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
        model_name=MODEL_NAME,
        max_checkpoints=1,
        max_length=MAX_LENGTHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHES,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        random_seed=RANDOM_SEED,
        gamma=GAMMA,
        temperature=TEMPERATURE,
        m0=M0,
        s=S,
        momentum=MOMENTUM,
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

    train_dataset = VulnerabilityDataset(train_data, config)
    val_dataset = VulnerabilityDataset(val_data, config)
    test_dataset = VulnerabilityDataset(test_data, config)

    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
    )
    log.print(f"Total classes: {trainer.num_classes}")
    trainer.train()


if __name__ == "__main__":
    main()
