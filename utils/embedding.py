from typing import Dict, List, Tuple
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class EmbeddingExtractor:
    @staticmethod
    def extract_embeddings(
        model: nn.Module,
        dataloader: DataLoader,
        class_to_idx: Dict[Tuple[bool, str], int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[int]]:
        embeddings = []
        labels = []
        model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                embs = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device)
                )
                embeddings.append(embs.cpu())
                label_indices = [class_to_idx[k] for k in batch["class_key"]]
                labels.extend(label_indices)

        return torch.cat(embeddings, dim=0), labels
