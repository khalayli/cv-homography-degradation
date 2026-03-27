from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HPatchesPair:
    scene_name: str
    split: str
    pair_index: int
    image0_path: str
    image1_path: str
    homography_path: str


class HPatchesDataset:
    """
    Lightweight HPatches dataset loader.

    Expected scene layout (HPatches sequences):
      scene_name/
        1.ppm
        2.ppm
        ...
        6.ppm
        H_1_2
        H_1_3
        ...
        H_1_6

    Each sample pairs image 1 with image i (i in [2..6]).
    """

    def __init__(self, root: str, pairs_per_scene: int = 5) -> None:
        print("[HPatchesDataset.__init__] Initializing HPatches dataset...")
        print(f"[HPatchesDataset.__init__] root={root}")
        print(f"[HPatchesDataset.__init__] pairs_per_scene={pairs_per_scene}")

        self.root = str(Path(root).expanduser().resolve())
        self.pairs_per_scene = int(pairs_per_scene)
        self.pairs: List[HPatchesPair] = []

        self._build_index()

    def _build_index(self) -> None:
        print("[HPatchesDataset._build_index] Building dataset index...")

        root_path = Path(self.root)
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"HPatches root not found: {self.root}")

        scene_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
        print(f"[HPatchesDataset._build_index] Found {len(scene_dirs)} scene directories.")

        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            split = self._infer_split(scene_name)

            image1_path = scene_dir / "1.ppm"
            if not image1_path.exists():
                print(f"[HPatchesDataset._build_index] Missing 1.ppm in {scene_dir}, skipping.")
                continue

            pair_indices = self._select_pair_indices()

            for pair_index in pair_indices:
                image2_path = scene_dir / f"{pair_index}.ppm"
                homography_path = scene_dir / f"H_1_{pair_index}"

                if not image2_path.exists() or not homography_path.exists():
                    print(
                        "[HPatchesDataset._build_index] "
                        f"Missing pair files for scene={scene_name}, pair={pair_index}, "
                        f"image2_exists={image2_path.exists()}, "
                        f"H_exists={homography_path.exists()} -> skipping"
                    )
                    continue

                self.pairs.append(
                    HPatchesPair(
                        scene_name=scene_name,
                        split=split,
                        pair_index=int(pair_index),
                        image0_path=str(image1_path),
                        image1_path=str(image2_path),
                        homography_path=str(homography_path),
                    )
                )

        print(f"[HPatchesDataset._build_index] Total pairs indexed: {len(self.pairs)}")

    def _select_pair_indices(self) -> List[int]:
        # HPatches pairs are 1->(2..6)
        max_pairs = 5
        count = int(self.pairs_per_scene)
        count = max(1, min(count, max_pairs))
        return list(range(2, 2 + count))

    def _infer_split(self, scene_name: str) -> str:
        # HPatches naming convention: "i_*" or "v_*"
        if scene_name.startswith("i_"):
            return "illumination"
        if scene_name.startswith("v_"):
            return "viewpoint"
        return "unknown"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        pair = self.pairs[idx]

        print(f"[HPatchesDataset.__getitem__] Loading pair idx={idx}, scene={pair.scene_name}")

        image0 = self._read_image(pair.image0_path)
        image1 = self._read_image(pair.image1_path)
        H_gt = self._read_homography(pair.homography_path)

        # Return a dict compatible with run_experiment.extract_* helpers
        example = {
            "image0": image0,
            "image1": image1,
            "homography_gt": H_gt,
            "scene_name": pair.scene_name,
            "split": pair.split,
            "pair_index": pair.pair_index,
            "pair": pair,
        }

        return example

    def _read_image(self, path: str) -> np.ndarray:
        print(f"[HPatchesDataset._read_image] Reading image: {path}")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def _read_homography(self, path: str) -> np.ndarray:
        print(f"[HPatchesDataset._read_homography] Reading homography: {path}")
        H = np.loadtxt(path).astype(np.float32)
        if H.shape != (3, 3):
            raise ValueError(f"Homography has unexpected shape {H.shape} at {path}")
        return H

    def describe(self) -> Dict[str, object]:
        print("[HPatchesDataset.describe] Summarizing dataset...")
        summary = self._summarize()
        print(f"[HPatchesDataset.describe] summary={summary}")
        return summary

    def summarize(self) -> Dict[str, object]:
        print("[HPatchesDataset.summarize] Summarizing dataset...")
        summary = self._summarize()
        print(f"[HPatchesDataset.summarize] summary={summary}")
        return summary

    def _summarize(self) -> Dict[str, object]:
        splits: Dict[str, int] = {}
        scenes: Dict[str, int] = {}

        for pair in self.pairs:
            splits[pair.split] = splits.get(pair.split, 0) + 1
            scenes[pair.scene_name] = scenes.get(pair.scene_name, 0) + 1

        return {
            "root": self.root,
            "num_pairs": len(self.pairs),
            "num_scenes": len(scenes),
            "pairs_per_scene": self.pairs_per_scene,
            "split_counts": splits,
        }
