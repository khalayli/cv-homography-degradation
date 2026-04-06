import cv2 as cv
import numpy as np


class ProposedMatcher:
    """
    Degradation-aware matcher wrapper.

    Main behavior:
    - run a configured primary backend
    - compute lightweight degradation indicators
    - adapt fallback trigger + geometry settings from indicators
    - optionally run a fallback backend
    - return a normalized project-wide result dictionary
    """

    def __init__(self, cfg=None):
        print("[ProposedMatcher.__init__] Initializing proposed matcher...")
        print(f"[ProposedMatcher.__init__] cfg={cfg}")

        self.cfg = cfg or {}
        self.name = "proposed"

        self.proposed_cfg = {}
        self.primary_matcher = None
        self.fallback_matcher = None

        self._parse_config()

        try:
            self.primary_matcher = self._build_matcher(self.primary_name)
        except Exception as exc:
            print(f"[ProposedMatcher.__init__] Primary matcher init failed: {exc}")
            self.primary_matcher = None

        try:
            self.fallback_matcher = self._build_matcher(self.fallback_name) if self.fallback_enabled else None
        except Exception as exc:
            print(f"[ProposedMatcher.__init__] Fallback matcher init failed: {exc}")
            self.fallback_matcher = None

    def _parse_config(self):
        print("[ProposedMatcher._parse_config] Parsing proposed matcher config...")

        self.proposed_cfg = dict(self.cfg.get("proposed", {}))

        self.primary_name = str(self.proposed_cfg.get("primary_matcher", "orb")).lower()
        self.fallback_name = str(self.proposed_cfg.get("fallback_matcher", "orb")).lower()
        self.fallback_enabled = bool(self.proposed_cfg.get("fallback_enabled", True))
        self.adaptive_thresholds = bool(self.proposed_cfg.get("adaptive_thresholds", True))
        self.min_matches = int(self.proposed_cfg.get("min_matches_before_fallback", 30))

        # Degradation thresholds (tunable through config)
        self.blur_low_threshold = float(self.proposed_cfg.get("blur_low_threshold", 60.0))
        self.noise_high_threshold = float(self.proposed_cfg.get("noise_high_threshold", 12.0))
        self.contrast_low_threshold = float(self.proposed_cfg.get("contrast_low_threshold", 25.0))

        # Adaptation limits
        self.max_adaptive_min_matches = int(self.proposed_cfg.get("max_adaptive_min_matches", 60))
        self.reproj_threshold_nominal = float(self.proposed_cfg.get("reproj_threshold_nominal", 3.0))
        self.reproj_threshold_hard = float(self.proposed_cfg.get("reproj_threshold_hard", 5.0))
        self.reproj_threshold_very_hard = float(self.proposed_cfg.get("reproj_threshold_very_hard", 6.0))

        print(f"[ProposedMatcher._parse_config] primary={self.primary_name}, fallback={self.fallback_name}")
        print(f"[ProposedMatcher._parse_config] fallback_enabled={self.fallback_enabled}")
        print(f"[ProposedMatcher._parse_config] min_matches={self.min_matches}")

    def _build_matcher(self, matcher_name):
        print(f"[ProposedMatcher._build_matcher] Building matcher={matcher_name}")

        matcher_name = str(matcher_name).lower()
        if matcher_name == "orb":
            from .orb import ORBMatcher

            return ORBMatcher(self.cfg)
        if matcher_name == "xfeat":
            from .xfeat import XFeatMatcher

            return XFeatMatcher(self.cfg)

        raise ValueError(f"Unknown matcher backend: {matcher_name}")

    def _validate_image(self, image, image_name):
        if image is None:
            raise ValueError(f"{image_name} is None")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{image_name} must be a numpy array")
        if image.size == 0:
            raise ValueError(f"{image_name} is empty")

    def _to_gray_u8(self, image):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 3 and image.shape[2] == 3:
            return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if image.ndim == 2:
            return image
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def _estimate_one_image_indicators(self, gray):
        lap = cv.Laplacian(gray, cv.CV_32F, ksize=3)
        blur_score = float(np.var(lap))

        med = cv.medianBlur(gray, 3)
        residual = gray.astype(np.float32) - med.astype(np.float32)
        noise_score = float(np.std(residual))

        contrast_score = float(np.std(gray))
        brightness_score = float(np.mean(gray))

        q95 = np.quantile(gray, 0.95)
        q05 = np.quantile(gray, 0.05)
        dynamic_range = float(q95 - q05)

        # Blockiness proxy: JPEG artifacts typically increase 8x8 boundary jumps.
        if gray.shape[0] > 8 and gray.shape[1] > 8:
            v_diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
            h_diff = np.abs(np.diff(gray.astype(np.float32), axis=0))

            v_boundary = v_diff[:, 7::8] if v_diff.shape[1] >= 8 else v_diff
            h_boundary = h_diff[7::8, :] if h_diff.shape[0] >= 8 else h_diff
            v_non = np.delete(v_diff, np.arange(7, v_diff.shape[1], 8), axis=1) if v_diff.shape[1] >= 8 else v_diff
            h_non = np.delete(h_diff, np.arange(7, h_diff.shape[0], 8), axis=0) if h_diff.shape[0] >= 8 else h_diff

            boundary_energy = 0.5 * (
                float(np.mean(v_boundary)) + float(np.mean(h_boundary))
            )
            non_boundary_energy = 0.5 * (
                float(np.mean(v_non)) + float(np.mean(h_non))
            )
            jpeg_score = float(max(boundary_energy - non_boundary_energy, 0.0))
        else:
            jpeg_score = 0.0

        return {
            "blur_score": blur_score,
            "noise_score": noise_score,
            "contrast_score": contrast_score,
            "brightness_score": brightness_score,
            "dynamic_range_score": dynamic_range,
            "jpeg_score": jpeg_score,
        }

    def _estimate_degradation_indicators(self, image0, image1):
        print("[ProposedMatcher._estimate_degradation_indicators] Estimating indicators...")

        gray0 = self._to_gray_u8(image0)
        gray1 = self._to_gray_u8(image1)

        i0 = self._estimate_one_image_indicators(gray0)
        i1 = self._estimate_one_image_indicators(gray1)

        # Use worst-case values because matching depends on both images.
        indicators = {
            "blur_score": float(min(i0["blur_score"], i1["blur_score"])),
            "noise_score": float(max(i0["noise_score"], i1["noise_score"])),
            "contrast_score": float(min(i0["contrast_score"], i1["contrast_score"])),
            "brightness_score": float(0.5 * (i0["brightness_score"] + i1["brightness_score"])),
            "dynamic_range_score": float(min(i0["dynamic_range_score"], i1["dynamic_range_score"])),
            "jpeg_score": float(max(i0["jpeg_score"], i1["jpeg_score"])),
        }

        print(f"[ProposedMatcher._estimate_degradation_indicators] indicators={indicators}")
        return indicators

    def _adapt_matcher_settings(self, indicators):
        print("[ProposedMatcher._adapt_matcher_settings] Adapting settings...")
        print(f"[ProposedMatcher._adapt_matcher_settings] indicators={indicators}")

        adapted_min_matches = int(self.min_matches)
        hard_degradation = False
        very_hard_degradation = False

        if indicators["blur_score"] < self.blur_low_threshold:
            adapted_min_matches += 8
            hard_degradation = True
        if indicators["noise_score"] > self.noise_high_threshold:
            adapted_min_matches += 8
            hard_degradation = True
        if indicators["contrast_score"] < self.contrast_low_threshold:
            adapted_min_matches += 6
            hard_degradation = True
        if indicators["jpeg_score"] > 2.5:
            adapted_min_matches += 4
            hard_degradation = True

        # Strong combined degradation, use the loosest geometry and easiest fallback trigger.
        if indicators["blur_score"] < 0.5 * self.blur_low_threshold and indicators["noise_score"] > 1.5 * self.noise_high_threshold:
            very_hard_degradation = True

        adapted_min_matches = min(adapted_min_matches, self.max_adaptive_min_matches)

        return {
            "min_matches_before_fallback": adapted_min_matches,
            "hard_degradation": hard_degradation,
            "very_hard_degradation": very_hard_degradation,
        }

    def _suggest_geom_overrides(self, adaptation):
        print("[ProposedMatcher._suggest_geom_overrides] Suggesting geometry overrides...")
        print(f"[ProposedMatcher._suggest_geom_overrides] adaptation={adaptation}")

        if not self.adaptive_thresholds:
            return {}

        if adaptation.get("very_hard_degradation", False):
            return {
                "reproj_threshold": float(self.reproj_threshold_very_hard),
                "confidence": 0.999,
                "max_iters": 8000,
            }

        if adaptation.get("hard_degradation", False):
            return {
                "reproj_threshold": float(self.reproj_threshold_hard),
                "confidence": 0.999,
                "max_iters": 6000,
            }

        return {
            "reproj_threshold": float(self.reproj_threshold_nominal),
            "confidence": 0.999,
            "max_iters": 5000,
        }

    def _run_primary(self, image0, image1):
        print("[ProposedMatcher._run_primary] Running primary matcher...")
        if self.primary_matcher is None:
            return None
        try:
            return self.primary_matcher.match(image0, image1)
        except Exception as exc:
            print(f"[ProposedMatcher._run_primary] Primary matcher failed: {exc}")
            return None

    def _run_fallback(self, image0, image1):
        print("[ProposedMatcher._run_fallback] Running fallback matcher...")
        if not self.fallback_enabled or self.fallback_matcher is None:
            return None
        try:
            return self.fallback_matcher.match(image0, image1)
        except Exception as exc:
            print(f"[ProposedMatcher._run_fallback] Fallback matcher failed: {exc}")
            return None

    def _should_fallback(self, primary_result, adaptation):
        print("[ProposedMatcher._should_fallback] Deciding whether to fallback...")
        if primary_result is None:
            return True

        effective_min_matches = int(adaptation.get("min_matches_before_fallback", self.min_matches))
        num_matches = int(primary_result.get("num_matches", 0))

        print(
            f"[ProposedMatcher._should_fallback] num_matches={num_matches}, "
            f"effective_min_matches={effective_min_matches}"
        )
        return num_matches < effective_min_matches

    def _normalize_points(self, pts):
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)

        arr = np.asarray(pts, dtype=np.float32)
        arr = np.squeeze(arr)

        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                return np.zeros((0, 2), dtype=np.float32)
            arr = arr.reshape(1, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.float32)
        return arr.astype(np.float32)

    def _normalize_match_result(self, raw_result, backend_used, fallback_used, geom_overrides):
        print("[ProposedMatcher._normalize_match_result] Normalizing result...")
        if raw_result is None:
            return self._build_empty_result(backend_used, fallback_used, geom_overrides)

        pts0 = self._normalize_points(raw_result.get("matched_points0"))
        pts1 = self._normalize_points(raw_result.get("matched_points1"))

        n = int(min(len(pts0), len(pts1)))
        pts0 = pts0[:n]
        pts1 = pts1[:n]

        scores = raw_result.get("scores", None)
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)[:n]

        return {
            "matched_points0": pts0,
            "matched_points1": pts1,
            "num_matches": n,
            "scores": scores,
            "backend_used": str(backend_used),
            "fallback_used": int(bool(fallback_used)),
            "geom_overrides": dict(geom_overrides or {}),
        }

    def _build_empty_result(self, backend_used="none", fallback_used=0, geom_overrides=None):
        if geom_overrides is None:
            geom_overrides = {}
        return {
            "matched_points0": np.zeros((0, 2), dtype=np.float32),
            "matched_points1": np.zeros((0, 2), dtype=np.float32),
            "num_matches": 0,
            "scores": None,
            "backend_used": str(backend_used),
            "fallback_used": int(bool(fallback_used)),
            "geom_overrides": dict(geom_overrides),
        }

    def match(self, image0, image1):
        print("[ProposedMatcher.match] Matching image pair with proposed method...")
        print(f"[ProposedMatcher.match] image0_shape={getattr(image0, 'shape', None)}")
        print(f"[ProposedMatcher.match] image1_shape={getattr(image1, 'shape', None)}")

        self._validate_image(image0, "image0")
        self._validate_image(image1, "image1")

        indicators = self._estimate_degradation_indicators(image0, image1)
        adaptation = self._adapt_matcher_settings(indicators)
        geom_overrides = self._suggest_geom_overrides(adaptation)

        primary_result = self._run_primary(image0, image1)

        if self.fallback_enabled and self._should_fallback(primary_result, adaptation):
            fallback_result = self._run_fallback(image0, image1)
            if fallback_result is not None:
                return self._normalize_match_result(
                    fallback_result,
                    backend_used=self.fallback_name,
                    fallback_used=1,
                    geom_overrides=geom_overrides,
                )

        return self._normalize_match_result(
            primary_result,
            backend_used=self.primary_name,
            fallback_used=0,
            geom_overrides=geom_overrides,
        )


def build_proposed_matcher(cfg=None):
    print("[build_proposed_matcher] Building proposed matcher...")
    print(f"[build_proposed_matcher] cfg={cfg}")
    matcher = ProposedMatcher(cfg)
    print("[build_proposed_matcher] Proposed matcher built successfully")
    return matcher
