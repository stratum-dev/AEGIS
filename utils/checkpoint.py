import glob
import os
import torch
from utils.config import ModelConfig


def list_checkpoints(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "pareto_checkpoint_epoch_*.pth"))

    return list(
        map(
            lambda checkpoint: int(
                os.path.basename(checkpoint)
                .replace("pareto_checkpoint_epoch_", "")
                .replace(".pth", "")
            ),
            checkpoints,
        )
    )


def save_checkpoint_with_limit(
    model,
    avg_proto,
    geo_proto,
    weight_proto,
    class_to_idx,
    idx_to_class,
    model_config: ModelConfig,
    epoch,
    output_dir,
    max_checkpoints=20,
):
    checkpoint_path = os.path.join(output_dir, f"pareto_checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "train_avg_prototypes": avg_proto.cpu(),
            "train_geo_prototypes": geo_proto.cpu(),
            "train_weight_prototypes": weight_proto.cpu(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "epoch": epoch,
        },
        checkpoint_path,
    )

    checkpoints = glob.glob(os.path.join(output_dir, "pareto_checkpoint_epoch_*.pth"))

    if len(checkpoints) > max_checkpoints:

        def extract_epoch(path):
            basename = os.path.basename(path)
            try:
                return int(
                    basename.replace("pareto_checkpoint_epoch_", "").replace(".pth", "")
                )
            except ValueError:
                return -1

        checkpoints.sort(key=extract_epoch)
        os.remove(checkpoints[0])
