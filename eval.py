import warnings
from utils.config import EvalConfig
from utils.evaluator import Evaluator
from utils.file import list_all_subdirs
from utils.checkpoint import list_checkpoints
import os
from InquirerPy import inquirer

warnings.filterwarnings("ignore")

MODEL_DIR = inquirer.select(
    message="Select the model directory: ",
    choices=list_all_subdirs(os.path.join(".", "models")),
).execute()
CHECKPOINT = inquirer.select(
    message="Specify the epoch of checkpoint: ", choices=list_checkpoints(MODEL_DIR)
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
