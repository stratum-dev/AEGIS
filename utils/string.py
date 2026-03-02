def print_dict_pipe(d: dict, precision=5):
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            v = f"{v:.{precision}f}"
        parts.append(f"{k}={v}")
    print(" | ".join(parts))
