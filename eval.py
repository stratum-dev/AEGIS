import argparse
import warnings
import torch
from utils.config import EvalConfig
from utils.evaluator import Evaluator
from utils.seed import set_seed
from utils.file import list_all_subdirs
import os
from InquirerPy import inquirer

warnings.filterwarnings("ignore")

MODEL_DIR = inquirer.select(
    message="Select the model directory: ",
    choices=list_all_subdirs(os.path.join(".", "models")),
).execute()
CHECKPOINT = inquirer.number(
    message="Specify the epoch of checkpoint: ", float_allowed=False
).execute()
BATCH_SIZE = inquirer.number(
    message="Batch size for evaluation: ", min_allowed=1, float_allowed=False
).execute()
DEVICE = inquirer.text(message="Device:", default="cuda").execute()


def main():
    config = EvalConfig(
        batch_size=int(BATCH_SIZE),
        model_dir=MODEL_DIR,
        checkpoint=CHECKPOINT,
        device=DEVICE,
    )

    evaluator = Evaluator(config)
    evaluator.evaluate()
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
