# src/matchers/xfeat.py

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple
import os
import cv2
import numpy as np


class XFeatMatcher:
    """
    Robust wrapper for XFeat.

    Purpose:
    - Load or initialize the XFeat model
    - Accept two images
    - Run feature extraction + matching
    - Return matched keypoints in a simple project-wide format

    Expected usage in the project:
        matcher = XFeatMatcher(cfg)
        result = matcher.match(image0, image1)

    Expected return format from match(...):
        {
            "matched_points0": <Nx2 array>,
            "matched_points1": <Nx2 array>,
            "num_matches": <int>,
            "scores": <optional match scores or None>,
        }

    Notes:
    - Keeps output format consistent with ORB and other matchers.
    - Does not estimate homography.
    - Does not compute metrics.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        print("[XFeatMatcher.__init__] Initializing XFeat matcher...")
        print(f"[XFeatMatcher.__init__] cfg={cfg}")

        self.cfg = cfg or {}
        self.name = "xfeat"

        self.xfeat_cfg: Dict[str, Any] = dict(self.cfg.get("xfeat", {}))
        self.device: str = str(self.xfeat_cfg.get("device", "cpu"))
        self.top_k: int = int(self.xfeat_cfg.get("top_k", 2000))
        self.weights_path: Optional[str] = self.xfeat_cfg.get("weights_path", None)
        self.score_threshold: Optional[float] = self.xfeat_cfg.get("score_threshold", None)

        self._torch = None
        self._model = None
        self._backend_name = "xfeat"
        self._assert_required_weights_present()

        self._check_dependencies()
        self._model = self._load_model()

        print(f"[XFeatMatcher.__init__] device={self.device}")
        print(f"[XFeatMatcher.__init__] top_k={self.top_k}")
        print(f"[XFeatMatcher.__init__] weights_path={self.weights_path}")
        print(f"[XFeatMatcher.__init__] score_threshold={self.score_threshold}")
        print("[XFeatMatcher.__init__] Done.")

    def _check_dependencies(self) -> None:
        print("[XFeatMatcher._check_dependencies] Checking XFeat dependencies...")

        try:
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError(
                "XFeat requires PyTorch, but 'torch' is not installed in this environment."
            ) from exc

        print("[XFeatMatcher._check_dependencies] torch import OK")

    def _assert_required_weights_present(self) -> None:
        print("[XFeatMatcher._assert_required_weights_present] Checking configured weights...")

        print(f"[XFeatMatcher._assert_required_weights_present] weights_path={self.weights_path}")

        if self.weights_path is None or str(self.weights_path).strip() in {"", "null", "None"}:
            raise ValueError(
                "[XFeatMatcher._assert_required_weights_present] "
                "xfeat.weights_path is null/empty. Refusing to initialize XFeat without pretrained weights."
            )

        if not os.path.isfile(str(self.weights_path)):
            raise FileNotFoundError(
                "[XFeatMatcher._assert_required_weights_present] "
                f"Expected XFeat weights file does not exist: {self.weights_path}"
            )

        print("[XFeatMatcher._assert_required_weights_present] Weight check passed.")
    def _load_model(self):
        print("[XFeatMatcher._load_model] Loading XFeat model...")

        # Try a few common import paths so the wrapper is flexible.
        candidate_imports = [
            ("modules.xfeat", "XFeat"),
            ("xfeat", "XFeat"),
            ("accelerated_features.modules.xfeat", "XFeat"),
            ("src.matchers.xfeat_model", "XFeat"),
        ]

        last_error = None

        for module_name, class_name in candidate_imports:
            try:
                print(f"[XFeatMatcher._load_model] Trying import {module_name}.{class_name}")
                module = importlib.import_module(module_name)
                model_cls = getattr(module, class_name)
                model = self._instantiate_model(model_cls)
                self._backend_name = f"{module_name}.{class_name}"
                print(f"[XFeatMatcher._load_model] Loaded backend: {self._backend_name}")
                return model
            except Exception as exc:
                print(f"[XFeatMatcher._load_model] Failed import/init for {module_name}.{class_name}: {exc}")
                last_error = exc

        raise ImportError(
            "Could not import or initialize XFeat. "
            "Make sure the XFeat package/code is installed and accessible in the environment."
        ) from last_error

    def _instantiate_model(self, model_cls):
        print("[XFeatMatcher._instantiate_model] Instantiating model...")
        print(f"[XFeatMatcher._instantiate_model] model_cls={model_cls}")
        print(f"[XFeatMatcher._instantiate_model] weights_path={self.weights_path}")
        print(f"[XFeatMatcher._instantiate_model] top_k={self.top_k}")
        print(f"[XFeatMatcher._instantiate_model] device={self.device}")

        if self.weights_path is None or str(self.weights_path).strip() in {"", "null", "None"}:
            raise ValueError(
                "[XFeatMatcher._instantiate_model] "
                "weights_path is null/empty. Refusing permissive constructor fallback."
            )

        constructor_attempts = [
            (
                "weights+top_k",
                lambda: model_cls(weights=self.weights_path, top_k=self.top_k),
            ),
            (
                "weights_path+top_k",
                lambda: model_cls(weights_path=self.weights_path, top_k=self.top_k),
            ),
            (
                "weights_only",
                lambda: model_cls(weights=self.weights_path),
            ),
            (
                "weights_path_only",
                lambda: model_cls(weights_path=self.weights_path),
            ),
        ]

        last_error = None
        model = None
        used_attempt_name = None

        for attempt_idx, (attempt_name, attempt_fn) in enumerate(constructor_attempts):
            try:
                print(
                    "[XFeatMatcher._instantiate_model] "
                    f"constructor attempt {attempt_idx}: {attempt_name}"
                )
                model = attempt_fn()
                used_attempt_name = attempt_name
                print(
                    "[XFeatMatcher._instantiate_model] "
                    f"constructor succeeded with {attempt_name}"
                )
                break
            except Exception as exc:
                print(
                    "[XFeatMatcher._instantiate_model] "
                    f"constructor attempt {attempt_idx} ({attempt_name}) failed: {exc}"
                )
                last_error = exc

        if model is None:
            raise RuntimeError(
                "[XFeatMatcher._instantiate_model] Failed to instantiate XFeat model "
                "with an explicit weights path."
            ) from last_error

        if hasattr(model, "top_k"):
            try:
                model.top_k = self.top_k
                print(f"[XFeatMatcher._instantiate_model] Set model.top_k={self.top_k}")
            except Exception as exc:
                print(
                    "[XFeatMatcher._instantiate_model] "
                    f"warning: could not set model.top_k: {exc}"
                )

        if hasattr(model, "to"):
            try:
                model = model.to(self.device)
                print(f"[XFeatMatcher._instantiate_model] moved model to {self.device}")
            except Exception as exc:
                print(
                    "[XFeatMatcher._instantiate_model] "
                    f"warning: could not move model to device: {exc}"
                )

        if hasattr(model, "eval"):
            try:
                model.eval()
                print("[XFeatMatcher._instantiate_model] model.eval() done")
            except Exception as exc:
                print(
                    "[XFeatMatcher._instantiate_model] "
                    f"warning: model.eval() failed: {exc}"
                )

        print(
            "[XFeatMatcher._instantiate_model] "
            f"Finished model init using constructor={used_attempt_name}"
        )
        return model

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        print("[XFeatMatcher._preprocess_image] Preprocessing image...")
        print(f"[XFeatMatcher._preprocess_image] image_shape={getattr(image, 'shape', None)}")

        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if image.size == 0:
            raise ValueError("Input image is empty")

        # Ensure uint8.
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Keep as color unless grayscale was provided.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape for XFeat: {image.shape}")

        return image

    def _to_torch_image(self, image: np.ndarray):
        print("[XFeatMatcher._to_torch_image] Converting image to torch tensor...")

        # OpenCV loads BGR, most torch vision code expects RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self._torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

        if self.device:
            tensor = tensor.to(self.device)

        print(f"[XFeatMatcher._to_torch_image] tensor_shape={tuple(tensor.shape)}")
        return tensor

    def _run_model(self, image0: np.ndarray, image1: np.ndarray):
        print("[XFeatMatcher._run_model] Running XFeat model on image pair...")

        # Convert once here so we can support torch-based APIs.
        tensor0 = self._to_torch_image(image0)
        tensor1 = self._to_torch_image(image1)

        # Try several common APIs used by feature matchers.
        candidate_calls = [
            ("match_xfeat", lambda: self._model.match_xfeat(tensor0, tensor1, top_k=self.top_k)),
            ("match", lambda: self._model.match(tensor0, tensor1, top_k=self.top_k)),
            ("__call__", lambda: self._model(tensor0, tensor1)),
            ("detectAndCompute+match", lambda: self._run_detect_compute_api(tensor0, tensor1)),
        ]

        last_error = None

        with self._torch.no_grad():
            for api_name, api_call in candidate_calls:
                try:
                    if api_name == "match_xfeat" and not hasattr(self._model, "match_xfeat"):
                        continue
                    if api_name == "match" and not hasattr(self._model, "match"):
                        continue
                    if api_name == "__call__" and not callable(self._model):
                        continue
                    if api_name == "detectAndCompute+match" and not (
                        hasattr(self._model, "detectAndCompute") or hasattr(self._model, "detect_and_compute")
                    ):
                        continue

                    print(f"[XFeatMatcher._run_model] Trying API: {api_name}")
                    raw_output = api_call()
                    print(f"[XFeatMatcher._run_model] API {api_name} succeeded")
                    return raw_output
                except Exception as exc:
                    print(f"[XFeatMatcher._run_model] API {api_name} failed: {exc}")
                    last_error = exc

        raise RuntimeError(
            "XFeat model loaded, but no supported matching API succeeded."
        ) from last_error

    def _run_detect_compute_api(self, tensor0, tensor1):
        print("[XFeatMatcher._run_detect_compute_api] Trying detect/compute style API...")

        detect_fn = None
        if hasattr(self._model, "detectAndCompute"):
            detect_fn = self._model.detectAndCompute
        elif hasattr(self._model, "detect_and_compute"):
            detect_fn = self._model.detect_and_compute

        if detect_fn is None:
            raise AttributeError("Model has no detect/compute API")

        feats0 = detect_fn(tensor0)
        feats1 = detect_fn(tensor1)

        # If the model exposes a separate matcher method, use it.
        if hasattr(self._model, "match"):
            return self._model.match(feats0, feats1)

        # Otherwise return tuple and let parser try to interpret it.
        return (feats0, feats1)

    def _parse_model_output(self, raw_output):
        print("[XFeatMatcher._parse_model_output] Parsing raw XFeat output...")
        print(f"[XFeatMatcher._parse_model_output] raw_output_type={type(raw_output)}")

        # Case 1: dict output.
        if isinstance(raw_output, dict):
            result = self._parse_dict_output(raw_output)
            if result is not None:
                return result

        # Case 2: tuple/list output.
        if isinstance(raw_output, (tuple, list)):
            result = self._parse_sequence_output(raw_output)
            if result is not None:
                return result

        raise ValueError(
            "Unsupported XFeat output format. "
            "You may need to adapt _parse_model_output to your team's XFeat version."
        )

    def _parse_dict_output(self, raw_output: Dict[str, Any]):
        print("[XFeatMatcher._parse_dict_output] Trying dictionary parse...")

        # Common key variants.
        pts0_keys = ["matched_points0", "mkpts0", "keypoints0", "pts0"]
        pts1_keys = ["matched_points1", "mkpts1", "keypoints1", "pts1"]
        score_keys = ["scores", "confidence", "match_scores"]

        pts0 = self._find_first_present(raw_output, pts0_keys)
        pts1 = self._find_first_present(raw_output, pts1_keys)
        scores = self._find_first_present(raw_output, score_keys, default=None)

        if pts0 is None or pts1 is None:
            print("[XFeatMatcher._parse_dict_output] No direct matched point keys found")
            return None

        pts0 = self._to_numpy_points(pts0)
        pts1 = self._to_numpy_points(pts1)
        scores = self._to_numpy_scores(scores)

        return self._finalize_result(pts0, pts1, scores)

    def _parse_sequence_output(self, raw_output):
        print("[XFeatMatcher._parse_sequence_output] Trying sequence parse...")

        # Common possibilities:
        # - (mkpts0, mkpts1)
        # - (mkpts0, mkpts1, scores)
        if len(raw_output) >= 2:
            try:
                pts0 = self._to_numpy_points(raw_output[0])
                pts1 = self._to_numpy_points(raw_output[1])
                scores = self._to_numpy_scores(raw_output[2]) if len(raw_output) >= 3 else None
                return self._finalize_result(pts0, pts1, scores)
            except Exception as exc:
                print(f"[XFeatMatcher._parse_sequence_output] direct sequence parse failed: {exc}")

        return None

    def _find_first_present(self, data: Dict[str, Any], keys, default=None):
        for key in keys:
            if key in data:
                print(f"[XFeatMatcher._find_first_present] found key={key}")
                return data[key]
        return default

    def _to_numpy_points(self, value) -> np.ndarray:
        print(f"[XFeatMatcher._to_numpy_points] value_type={type(value)}")

        if value is None:
            return np.zeros((0, 2), dtype=np.float32)

        if self._torch is not None and hasattr(self._torch, "is_tensor") and self._torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        value = np.asarray(value, dtype=np.float32)

        # Squeeze singleton dimensions.
        value = np.squeeze(value)

        if value.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if value.ndim == 1:
            if value.shape[0] != 2:
                raise ValueError(f"Expected 2D point(s), got shape {value.shape}")
            value = value.reshape(1, 2)

        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"Expected Nx2 points, got shape {value.shape}")

        return value.astype(np.float32)

    def _to_numpy_scores(self, value):
        if value is None:
            return None

        print(f"[XFeatMatcher._to_numpy_scores] value_type={type(value)}")

        if self._torch is not None and hasattr(self._torch, "is_tensor") and self._torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        value = np.asarray(value, dtype=np.float32).reshape(-1)
        return value

    def _apply_top_k(self, pts0: np.ndarray, pts1: np.ndarray, scores):
        print("[XFeatMatcher._apply_top_k] Applying top_k if needed...")

        num_matches = min(len(pts0), len(pts1))
        if scores is not None:
            num_matches = min(num_matches, len(scores))

        if num_matches == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                None if scores is None else np.zeros((0,), dtype=np.float32),
            )

        pts0 = pts0[:num_matches]
        pts1 = pts1[:num_matches]
        if scores is not None:
            scores = scores[:num_matches]

        if self.score_threshold is not None and scores is not None:
            keep = scores >= float(self.score_threshold)
            pts0 = pts0[keep]
            pts1 = pts1[keep]
            scores = scores[keep]
            print(f"[XFeatMatcher._apply_top_k] after score_threshold count={len(pts0)}")

        if len(pts0) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                None if scores is None else np.zeros((0,), dtype=np.float32),
            )

        if scores is not None and len(scores) > self.top_k:
            order = np.argsort(-scores)[: self.top_k]
            pts0 = pts0[order]
            pts1 = pts1[order]
            scores = scores[order]
            print(f"[XFeatMatcher._apply_top_k] kept top {self.top_k} by score")
        elif len(pts0) > self.top_k:
            pts0 = pts0[: self.top_k]
            pts1 = pts1[: self.top_k]
            if scores is not None:
                scores = scores[: self.top_k]
            print(f"[XFeatMatcher._apply_top_k] truncated to top_k={self.top_k}")

        return pts0, pts1, scores

    def _finalize_result(self, pts0: np.ndarray, pts1: np.ndarray, scores):
        pts0, pts1, scores = self._apply_top_k(pts0, pts1, scores)

        result = {
            "matched_points0": pts0.astype(np.float32),
            "matched_points1": pts1.astype(np.float32),
            "num_matches": int(len(pts0)),
            "scores": scores,
        }

        print(f"[XFeatMatcher._finalize_result] num_matches={result['num_matches']}")
        return result

    def _build_empty_result(self):
        print("[XFeatMatcher._build_empty_result] Building empty result...")

        result = {
            "matched_points0": np.zeros((0, 2), dtype=np.float32),
            "matched_points1": np.zeros((0, 2), dtype=np.float32),
            "num_matches": 0,
            "scores": None,
        }

        print(f"[XFeatMatcher._build_empty_result] result={result}")
        return result

    def match(self, image0, image1):
        print("[XFeatMatcher.match] Matching image pair...")
        print(f"[XFeatMatcher.match] image0_shape={getattr(image0, 'shape', None)}")
        print(f"[XFeatMatcher.match] image1_shape={getattr(image1, 'shape', None)}")

        if image0 is None or image1 is None:
            raise ValueError("One or both input images are None")

        if not isinstance(image0, np.ndarray) or not isinstance(image1, np.ndarray):
            raise ValueError("One or both input images are not numpy arrays")

        if image0.size == 0 or image1.size == 0:
            raise ValueError("One or both input images are empty")

        image0 = self._preprocess_image(image0)
        image1 = self._preprocess_image(image1)

        raw_output = self._run_model(image0, image1)
        result = self._parse_model_output(raw_output)

        if result["num_matches"] == 0:
            print("[XFeatMatcher.match] No matches found")
            return self._build_empty_result()

        return result


def build_xfeat_matcher(cfg=None):
    print("[build_xfeat_matcher] Building XFeat matcher...")
    print(f"[build_xfeat_matcher] cfg={cfg}")
    matcher = XFeatMatcher(cfg)
    print("[build_xfeat_matcher] XFeat matcher built successfully")
    return matcher