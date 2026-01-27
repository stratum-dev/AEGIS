from typing import Dict, Any
import torch


class VulnerabilityDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

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
