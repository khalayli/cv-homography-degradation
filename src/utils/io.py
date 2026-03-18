import os
import json
from pathlib import Path
from typing import Any, Dict

import yaml

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"[ensure_dir] Ensured directory: {path}")

def read_yaml(path: str) -> Dict[str, Any]:
    print(f"[read_yaml] Reading config from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print("[read_yaml] Loaded keys:", list(data.keys()))
    return data

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    print(f"[write_json] Writing JSON to: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("[write_json] Done.")

def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    print(f"[write_text] Writing text to: {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("[write_text] Done.")
