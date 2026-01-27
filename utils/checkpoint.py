import glob
import os
import torch
from utils.aegis import AEGISModel
from utils.config import ModelConfig


def save_checkpoint_with_limit(
    model,
    avg_proto,
    geo_proto,
    weight_proto,
    class_to_idx,
    idx_to_class,
    config: ModelConfig,
    epoch,
    output_dir,
    max_checkpoints=20,
):
    checkpoint_path = os.path.join(output_dir, f"pareto_checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "train_avg_prototypes": avg_proto.cpu(),
            "train_geo_prototypes": geo_proto.cpu(),
            "train_weight_prototypes": weight_proto.cpu(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "epoch": epoch + 1,
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


def load_model_and_checkpoints(config: ModelConfig):
    """加载模型和检查点"""
    print(f"Loading model from: {config.MODEL_DIR}")
    model_path = os.path.join(
        config.MODEL_DIR, f"pareto_checkpoint_epoch_{config.CHECKPOINT}.pth"
    )
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    # 加载完整检查点
    print(f"Loading full checkpoint from: {model_path}")
    full_checkpoint = torch.load(model_path, map_location=config.DEVICE)

    # 从检查点重建类映射
    class_to_idx = full_checkpoint["class_to_idx"]
    idx_to_class = full_checkpoint["idx_to_class"]
    num_classes = len(class_to_idx)

    # 构建模型
    model = AEGISModel(
        config.MODEL_NAME,
        num_classes,
        config.M0,
        config.S,
        config.ROBERTA_LAYERS_TO_CONCAT,
    ).to(config.DEVICE)
    model.load_state_dict(full_checkpoint["model_state_dict"], strict=True)
    model.eval()

    # 加载原型
    geo_prototypes = full_checkpoint["train_geo_prototypes"].to(
        config.DEVICE
    )  # 已经归一化和分离
    print(f"Loaded geo_prototypes of shape: {geo_prototypes.shape}")

    return model, geo_prototypes, class_to_idx, idx_to_class
