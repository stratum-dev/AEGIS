import argparse
import warnings
import torch
from utils.config import EvalConfig
from utils.evaluator import Evaluator
from utils.seed import set_seed

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AEGIS model on a vulnerability dataset."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory of saved checkpoints (e.g., output_megavul_20260111-17-43-19)",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        required=True,
        # choices=[
        #     "bigvul",
        #     "megavul",
        #     "reposvul",
        #     "draper",
        #     "mvd",
        #     "devign",
        #     "reveal",
        #     "vuldeepecker",
        # ],
        help="Dataset subset name",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Checkpoint epoch number to evaluate (e.g., 15)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="codemetic/AEGIS",
        help="Hugging Face dataset name (default: codemetic/AEGIS)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="codemetic/CweBERT-mlm",
        help="Pretrained model name (default: codemetic/CweBERT-mlm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help='Device to use: "auto" (default), "cuda", or "cpu"',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 自动选择设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = EvalConfig(
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        model_dir=args.model_dir,
        subset_name=args.subset_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        checkpoint=args.checkpoint,
        device=device,
    )

    set_seed(config.RANDOM_SEED)
    evaluator = Evaluator(config)
    evaluator.evaluate()
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
