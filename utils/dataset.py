from typing import Dict, Any
import torch
from transformers import AutoTokenizer
from utils.config import ModelConfig


class VulnerabilityDataset(torch.utils.data.Dataset):
    def __init__(self, data, config: ModelConfig):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_REPO)
        # The UniCoderX and other early RoBERTa-based model does not specify the max_length, which may cause OOM. 
        # We set it to 2048 to prevent OOM, which is also the max length for most code models.
        self.max_length = min(self.tokenizer.model_max_length, 512)
        

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        source = str(item["source"])
        label = bool(item["label"])
        cwe = str(item["cwe"]) if label else "Non-vul"
        class_key = (label, cwe)

        inputs = self.tokenizer(
            source,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label,
            "cwe": cwe,
            "class_key": class_key,
        }


def custom_collate_fn(batch):
    """自定义批处理函数，防止非张量字段堆叠"""
    result = {}
    for key in batch[0]:
        if key in ["class_key", "cwe", "label"]:
            result[key] = [d[key] for d in batch]
        else:
            result[key] = torch.stack([d[key] for d in batch])
    return result
