import os
import random
import numpy as np

def seed_everything(seed: int) -> None:
    print(f"[seed_everything] Setting seed={seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print("[seed_everything] Done.")
