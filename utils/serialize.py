import json
import os
from typing import Any


def save_to_json(obj: Any, file_path: str, indent: int = 4) -> None:
    attrs = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        try:
            json.dumps(value)
            attrs[key] = value
        except (TypeError, ValueError):
            pass

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(attrs, f, indent=indent, ensure_ascii=False)
