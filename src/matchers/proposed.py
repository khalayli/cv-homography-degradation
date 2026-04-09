import cv2 as cv
import numpy as np
import os


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
        self.direction_profiles = {}
        self.active_direction_key = None
        self.active_direction_cfg = {}

        self.primary_matcher = None
        self.fallback_matcher = None

        self._parse_config()
        self._assert_required_xfeat_weights_present()

        self.primary_matcher = self._build_required_matcher(self.primary_name, role="primary")
        print(f"[ProposedMatcher.__init__] Primary matcher built: {self.primary_name}")

        if self.fallback_enabled:
            self.fallback_matcher = self._build_required_matcher(self.fallback_name, role="fallback")
            print(f"[ProposedMatcher.__init__] Fallback matcher built: {self.fallback_name}")
        else:
            print("[ProposedMatcher.__init__] Fallback disabled by config.")

    def _parse_config(self):
        print("[ProposedMatcher._parse_config] Parsing proposed matcher config...")

        self.proposed_cfg = dict(self.cfg.get("proposed", {}))
        self.primary_name = str(self.proposed_cfg.get("primary_matcher", "orb")).lower()
        self.fallback_name = str(self.proposed_cfg.get("fallback_matcher", "xfeat")).lower()
        self.fallback_enabled = bool(self.proposed_cfg.get("fallback_enabled", True))
        self.adaptive_thresholds = bool(self.proposed_cfg.get("adaptive_thresholds", True))

        valid_backends = {"orb", "xfeat"}
        if self.primary_name not in valid_backends:
            raise ValueError(f"Unsupported proposed.primary_matcher: {self.primary_name}")
        if self.fallback_enabled and self.fallback_name not in valid_backends:
            raise ValueError(f"Unsupported proposed.fallback_matcher: {self.fallback_name}")
        if self.fallback_enabled and self.primary_name == self.fallback_name:
            raise ValueError(
                "[ProposedMatcher._parse_config] primary_matcher and fallback_matcher must differ "
                "when fallback_enabled=True."
            )

        self.direction_profiles = dict(self.proposed_cfg.get("direction_profiles", {}))
        self.active_direction_key = self._get_direction_key()
        self.active_direction_cfg = dict(self.direction_profiles.get(self.active_direction_key, {}))

        print(
            "[ProposedMatcher._parse_config] "
            f"primary={self.primary_name}, fallback={self.fallback_name}, "
            f"fallback_enabled={self.fallback_enabled}, "
            f"active_direction_key={self.active_direction_key}"
        )
        print(
            "[ProposedMatcher._parse_config] "
            f"active_direction_cfg={self.active_direction_cfg}"
        )

        # Direction-aware fallback thresholds
        self.min_matches = int(self._profile_get("min_matches_before_fallback", 30))
        self.max_adaptive_min_matches = int(self._profile_get("max_adaptive_min_matches", 60))

        # Direction-aware severity increments
        self.min_matches_add_blur = int(self._profile_get("min_matches_add_blur", 24))
        self.min_matches_add_noise = int(self._profile_get("min_matches_add_noise", 24))
        self.min_matches_add_contrast = int(self._profile_get("min_matches_add_contrast", 12))
        self.min_matches_add_jpeg = int(self._profile_get("min_matches_add_jpeg", 12))
        self.min_matches_add_dynamic_range = int(self._profile_get("min_matches_add_dynamic_range", 16))
        self.min_matches_add_brightness = int(self._profile_get("min_matches_add_brightness", 8))

        # Direction-aware match quality thresholds
        self.fallback_min_spatial_coverage = float(
            self._profile_get("fallback_min_spatial_coverage", 0.015)
        )
        fallback_min_score_mean = self._profile_get("fallback_min_score_mean", 0.20)
        self.fallback_min_score_mean = (
            None if fallback_min_score_mean is None else float(fallback_min_score_mean)
        )

        # Direction-aware probe thresholds
        self.probe_min_inliers = int(self._profile_get("probe_min_inliers", 30))
        self.probe_min_inlier_ratio = float(self._profile_get("probe_min_inlier_ratio", 0.40))
        self.probe_min_inlier_coverage = float(self._profile_get("probe_min_inlier_coverage", 0.03))

        # Shared degradation indicator thresholds
        self.brightness_low_threshold = float(self.proposed_cfg.get("brightness_low_threshold", 45.0))
        self.brightness_high_threshold = float(self.proposed_cfg.get("brightness_high_threshold", 210.0))
        self.dynamic_range_low_threshold = float(self.proposed_cfg.get("dynamic_range_low_threshold", 55.0))
        self.jpeg_high_threshold = float(self.proposed_cfg.get("jpeg_high_threshold", 2.0))
        self.noise_flat_grad_threshold = float(self.proposed_cfg.get("noise_flat_grad_threshold", 12.0))

        self.blur_low_threshold = float(self.proposed_cfg.get("blur_low_threshold", 60.0))
        self.noise_high_threshold = float(self.proposed_cfg.get("noise_high_threshold", 12.0))
        self.contrast_low_threshold = float(self.proposed_cfg.get("contrast_low_threshold", 25.0))

        # Shared geometry override thresholds
        self.reproj_threshold_nominal = float(self.proposed_cfg.get("reproj_threshold_nominal", 3.0))
        self.reproj_threshold_hard = float(self.proposed_cfg.get("reproj_threshold_hard", 5.0))
        self.reproj_threshold_very_hard = float(self.proposed_cfg.get("reproj_threshold_very_hard", 6.0))

        # Shared cheap probe settings
        self.probe_reproj_threshold = float(self.proposed_cfg.get("probe_reproj_threshold", 3.0))
        self.probe_confidence = float(self.proposed_cfg.get("probe_confidence", 0.995))
        self.probe_max_iters = int(self.proposed_cfg.get("probe_max_iters", 2000))
        self.probe_min_area_ratio = float(self.proposed_cfg.get("probe_min_area_ratio", 0.20))
        self.probe_max_area_ratio = float(self.proposed_cfg.get("probe_max_area_ratio", 4.00))
        self.probe_margin_ratio = float(self.proposed_cfg.get("probe_margin_ratio", 0.25))

        print(
            "[ProposedMatcher._parse_config] "
            f"min_matches={self.min_matches}, max_adaptive_min_matches={self.max_adaptive_min_matches}"
        )
        print(
            "[ProposedMatcher._parse_config] "
            f"min_match_additions="
            f"blur:{self.min_matches_add_blur}, "
            f"noise:{self.min_matches_add_noise}, "
            f"contrast:{self.min_matches_add_contrast}, "
            f"jpeg:{self.min_matches_add_jpeg}, "
            f"dynamic_range:{self.min_matches_add_dynamic_range}, "
            f"brightness:{self.min_matches_add_brightness}"
        )
        print(
            "[ProposedMatcher._parse_config] "
            f"fallback_min_spatial_coverage={self.fallback_min_spatial_coverage}, "
            f"fallback_min_score_mean={self.fallback_min_score_mean}"
        )
        print(
            "[ProposedMatcher._parse_config] "
            f"probe_min_inliers={self.probe_min_inliers}, "
            f"probe_min_inlier_ratio={self.probe_min_inlier_ratio}, "
            f"probe_min_inlier_coverage={self.probe_min_inlier_coverage}"
        )

    def _assert_required_xfeat_weights_present(self):
        print("[ProposedMatcher._assert_required_xfeat_weights_present] Checking XFeat weight requirements...")

        xfeat_roles = []
        if self.primary_name == "xfeat":
            xfeat_roles.append("primary")
        if self.fallback_enabled and self.fallback_name == "xfeat":
            xfeat_roles.append("fallback")

        if not xfeat_roles:
            print("[ProposedMatcher._assert_required_xfeat_weights_present] XFeat not used by active strategy. Skipping check.")
            return

        xfeat_cfg = dict(self.cfg.get("xfeat", {}))
        weights_path = xfeat_cfg.get("weights_path", None)

        print(
            "[ProposedMatcher._assert_required_xfeat_weights_present] "
            f"xfeat_roles={xfeat_roles}, weights_path={weights_path}"
        )

        if weights_path is None or str(weights_path).strip() in {"", "null", "None"}:
            raise ValueError(
                "[ProposedMatcher._assert_required_xfeat_weights_present] "
                f"XFeat is required for roles={xfeat_roles}, but xfeat.weights_path is null/empty."
            )

        if not os.path.isfile(str(weights_path)):
            raise FileNotFoundError(
                "[ProposedMatcher._assert_required_xfeat_weights_present] "
                f"Expected XFeat weights file does not exist: {weights_path}"
            )

        print("[ProposedMatcher._assert_required_xfeat_weights_present] XFeat weights check passed.")

    def _get_direction_key(self):
        if not self.fallback_enabled:
            key = f"{self.primary_name}_only"
        else:
            key = f"{self.primary_name}_to_{self.fallback_name}"

        print(f"[ProposedMatcher._get_direction_key] key={key}")
        return key

    def _profile_get(self, key, default=None):
        if key in self.active_direction_cfg:
            value = self.active_direction_cfg[key]
            print(
                "[ProposedMatcher._profile_get] "
                f"Using direction-profile override for key='{key}': {value}"
            )
            return value

        value = self.proposed_cfg.get(key, default)
        print(
            "[ProposedMatcher._profile_get] "
            f"Using base proposed config for key='{key}': {value}"
        )
        return value

    def _build_required_matcher(self, matcher_name, role):
        print(
            "[ProposedMatcher._build_required_matcher] "
            f"Building required matcher for role={role}, matcher_name={matcher_name}"
        )
        try:
            matcher = self._build_matcher(matcher_name)
            print(
                "[ProposedMatcher._build_required_matcher] "
                f"Built matcher for role={role}, matcher_name={matcher_name}"
            )
            return matcher
        except Exception as exc:
            print(
                "[ProposedMatcher._build_required_matcher] "
                f"Failed to build matcher for role={role}, matcher_name={matcher_name}: {exc}"
            )
            raise RuntimeError(
                "[ProposedMatcher._build_required_matcher] "
                f"Failed to initialize required {role} matcher '{matcher_name}'."
            ) from exc

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

        # Estimate noise primarily in flat regions so the gradient threshold
        # actually influences the decision path. Fall back to the full-image
        # residual when the image has too few flat pixels.
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        grad_mag = cv.magnitude(gx, gy)
        flat_mask = grad_mag < float(self.noise_flat_grad_threshold)
        if int(np.count_nonzero(flat_mask)) >= 64:
            noise_score = float(np.std(residual[flat_mask]))
        else:
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
        severity_points = 0
        degradation_reasons = []

        blur_bad = indicators["blur_score"] < self.blur_low_threshold
        noise_bad = indicators["noise_score"] > self.noise_high_threshold
        contrast_bad = indicators["contrast_score"] < self.contrast_low_threshold
        jpeg_bad = indicators["jpeg_score"] > self.jpeg_high_threshold
        dynamic_range_bad = indicators["dynamic_range_score"] < self.dynamic_range_low_threshold
        brightness_bad = (
            indicators["brightness_score"] < self.brightness_low_threshold
            or indicators["brightness_score"] > self.brightness_high_threshold
        )

        if blur_bad:
            adapted_min_matches += self.min_matches_add_blur
            severity_points += 2
            degradation_reasons.append("blur")

        if noise_bad:
            adapted_min_matches += self.min_matches_add_noise
            severity_points += 2
            degradation_reasons.append("noise")

        if contrast_bad:
            adapted_min_matches += self.min_matches_add_contrast
            severity_points += 1
            degradation_reasons.append("contrast")

        if jpeg_bad:
            adapted_min_matches += self.min_matches_add_jpeg
            severity_points += 1
            degradation_reasons.append("jpeg")

        if dynamic_range_bad:
            adapted_min_matches += self.min_matches_add_dynamic_range
            severity_points += 2
            degradation_reasons.append("dynamic_range")

        if brightness_bad:
            adapted_min_matches += self.min_matches_add_brightness
            severity_points += 1
            degradation_reasons.append("brightness")

        hard_degradation = severity_points >= 1
        very_hard_degradation = (
            severity_points >= 4
            or (blur_bad and noise_bad)
            or (dynamic_range_bad and contrast_bad)
            or (jpeg_bad and contrast_bad)
        )

        adapted_min_matches = min(adapted_min_matches, self.max_adaptive_min_matches)

        adaptation = {
            "min_matches_before_fallback": adapted_min_matches,
            "hard_degradation": hard_degradation,
            "very_hard_degradation": very_hard_degradation,
            "severity_points": int(severity_points),
            "degradation_reasons": list(degradation_reasons),
            "active_direction_key": self.active_direction_key,
        }

        print(f"[ProposedMatcher._adapt_matcher_settings] adaptation={adaptation}")
        return adaptation

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

    def _compute_bbox_coverage(self, pts, image_shape):
        print("[ProposedMatcher._compute_bbox_coverage] Computing match coverage...")

        if image_shape is None or len(image_shape) < 2:
            print("[ProposedMatcher._compute_bbox_coverage] Invalid image shape. Returning 0.0")
            return 0.0

        pts = self._normalize_points(pts)
        h, w = int(image_shape[0]), int(image_shape[1])

        if len(pts) < 4 or h <= 0 or w <= 0:
            print(
                "[ProposedMatcher._compute_bbox_coverage] "
                f"len(pts)={len(pts)}, h={h}, w={w}. Returning 0.0"
            )
            return 0.0

        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))

        bbox_w = max(x_max - x_min, 0.0)
        bbox_h = max(y_max - y_min, 0.0)
        bbox_area = bbox_w * bbox_h
        image_area = float(h * w)

        coverage = float(bbox_area / image_area) if image_area > 0 else 0.0
        print(
            "[ProposedMatcher._compute_bbox_coverage] "
            f"bbox_area={bbox_area}, image_area={image_area}, coverage={coverage}"
        )
        return coverage

    # fallback should not be based only on raw count
    # this adds a simple geometric sanity signal:
    # are matches spread over the image or all clustered in one tiny patch?
    def _compute_match_quality(self, match_result, image0_shape, image1_shape):
        print("[ProposedMatcher._compute_match_quality] Computing match quality...")

        if match_result is None:
            quality = {
                "num_matches": 0,
                "score_mean": None,
                "coverage0": 0.0,
                "coverage1": 0.0,
                "spatial_coverage": 0.0,
            }
            print(f"[ProposedMatcher._compute_match_quality] quality={quality}")
            return quality

        pts0 = self._normalize_points(match_result.get("matched_points0"))
        pts1 = self._normalize_points(match_result.get("matched_points1"))
        n = int(min(len(pts0), len(pts1)))
        pts0 = pts0[:n]
        pts1 = pts1[:n]

        coverage0 = self._compute_bbox_coverage(pts0, image0_shape)
        coverage1 = self._compute_bbox_coverage(pts1, image1_shape)
        spatial_coverage = float(min(coverage0, coverage1))

        score_mean = None
        scores = match_result.get("scores", None)
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)[:n]
            if scores.size > 0:
                score_mean = float(np.mean(scores))

        quality = {
            "num_matches": n,
            "score_mean": score_mean,
            "coverage0": coverage0,
            "coverage1": coverage1,
            "spatial_coverage": spatial_coverage,
        }
        print(f"[ProposedMatcher._compute_match_quality] quality={quality}")
        return quality

    # Compute the area of a 2D polygon using the shoelace formula.
    # Used to measure the size of the warped image quadrilateral for homography sanity checks.
    def _polygon_area(self, pts):
        print("[ProposedMatcher._polygon_area] Computing polygon area...")

        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            print("[ProposedMatcher._polygon_area] Invalid polygon shape. Returning 0.0")
            return 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        area = float(area)

        print(f"[ProposedMatcher._polygon_area] area={area}")
        return area

    def _probe_primary_geometry(self, match_result, image0_shape, image1_shape):
        print("[ProposedMatcher._probe_primary_geometry] Probing primary geometry...")

        pts0 = self._normalize_points(match_result.get("matched_points0"))
        pts1 = self._normalize_points(match_result.get("matched_points1"))

        n = int(min(len(pts0), len(pts1)))
        pts0 = pts0[:n]
        pts1 = pts1[:n]

        probe = {
            "success": False,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "inlier_coverage0": 0.0,
            "inlier_coverage1": 0.0,
            "inlier_coverage": 0.0,
            "area_ratio": None,
            "corners_reasonable": False,
        }

        print(f"[ProposedMatcher._probe_primary_geometry] n={n}")

        if n < 4:
            print("[ProposedMatcher._probe_primary_geometry] Fewer than 4 matches. Probe failed.")
            return probe

        try:
            H, inlier_mask = cv.findHomography(
                pts0.reshape(-1, 1, 2),
                pts1.reshape(-1, 1, 2),
                method=cv.RANSAC,
                ransacReprojThreshold=float(self.probe_reproj_threshold),
                maxIters=int(self.probe_max_iters),
                confidence=float(self.probe_confidence),
            )
        except cv.error as exc:
            print(f"[ProposedMatcher._probe_primary_geometry] findHomography failed: {exc}")
            return probe

        if H is None or inlier_mask is None:
            print("[ProposedMatcher._probe_primary_geometry] H or inlier_mask is None.")
            return probe

        mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
        num_inliers = int(np.sum(mask))
        inlier_ratio = float(num_inliers / max(n, 1))

        inlier_pts0 = pts0[mask] if num_inliers > 0 else np.zeros((0, 2), dtype=np.float32)
        inlier_pts1 = pts1[mask] if num_inliers > 0 else np.zeros((0, 2), dtype=np.float32)

        inlier_coverage0 = self._compute_bbox_coverage(inlier_pts0, image0_shape) if num_inliers >= 4 else 0.0
        inlier_coverage1 = self._compute_bbox_coverage(inlier_pts1, image1_shape) if num_inliers >= 4 else 0.0
        inlier_coverage = float(min(inlier_coverage0, inlier_coverage1))

        h0, w0 = int(image0_shape[0]), int(image0_shape[1])
        h1, w1 = int(image1_shape[0]), int(image1_shape[1])

        corners0 = np.array(
            [
                [0.0, 0.0],
                [float(w0 - 1), 0.0],
                [float(w0 - 1), float(h0 - 1)],
                [0.0, float(h0 - 1)],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        area_ratio = None
        corners_reasonable = False

        try:
            warped_corners = cv.perspectiveTransform(corners0, H).reshape(-1, 2)

            finite_ok = bool(np.all(np.isfinite(warped_corners)))

            src_area = float(max(w0 * h0, 1))
            dst_area = self._polygon_area(warped_corners)
            area_ratio = float(dst_area / src_area)

            margin_x = float(self.probe_margin_ratio * w1)
            margin_y = float(self.probe_margin_ratio * h1)

            xs = warped_corners[:, 0]
            ys = warped_corners[:, 1]

            inside_loose_bounds = bool(
                np.all(xs > -margin_x)
                and np.all(xs < (w1 - 1 + margin_x))
                and np.all(ys > -margin_y)
                and np.all(ys < (h1 - 1 + margin_y))
            )

            corners_reasonable = bool(
                finite_ok
                and inside_loose_bounds
                and (self.probe_min_area_ratio <= area_ratio <= self.probe_max_area_ratio)
            )

            print(
                "[ProposedMatcher._probe_primary_geometry] "
                f"warped_corners={warped_corners.tolist()}, "
                f"area_ratio={area_ratio}, corners_reasonable={corners_reasonable}"
            )

        except cv.error as exc:
            print(f"[ProposedMatcher._probe_primary_geometry] perspectiveTransform failed: {exc}")

        probe = {
            "success": True,
            "num_inliers": int(num_inliers),
            "inlier_ratio": float(inlier_ratio),
            "inlier_coverage0": float(inlier_coverage0),
            "inlier_coverage1": float(inlier_coverage1),
            "inlier_coverage": float(inlier_coverage),
            "area_ratio": area_ratio,
            "corners_reasonable": bool(corners_reasonable),
        }

        print(f"[ProposedMatcher._probe_primary_geometry] probe={probe}")
        return probe

    # Fall back when the primary result looks weak either before geometry
    # (few matches / poor coverage / low confidence) or after a cheap
    # geometry probe (too few inliers / poor inlier coverage / implausible warp).
    def _should_fallback(self, primary_result, adaptation, image0_shape, image1_shape):
        print("[ProposedMatcher._should_fallback] Deciding whether to fallback...")

        if primary_result is None:
            print("[ProposedMatcher._should_fallback] primary_result is None -> fallback=True")
            return True

        effective_min_matches = int(
            adaptation.get("min_matches_before_fallback", self.min_matches)
        )

        quality = self._compute_match_quality(primary_result, image0_shape, image1_shape)

        num_matches = int(quality["num_matches"])
        spatial_coverage = float(quality["spatial_coverage"])
        score_mean = quality["score_mean"]

        effective_min_spatial_coverage = float(self.fallback_min_spatial_coverage)
        if adaptation.get("hard_degradation", False):
            effective_min_spatial_coverage *= 0.75

        count_too_low = num_matches < effective_min_matches
        coverage_too_low = num_matches >= 4 and spatial_coverage < effective_min_spatial_coverage

        score_check_enabled = self.fallback_min_score_mean is not None
        score_too_low = (
            score_check_enabled
            and score_mean is not None
            and num_matches >= 4
            and score_mean < float(self.fallback_min_score_mean)
        )

        print(
            "[ProposedMatcher._should_fallback] "
            f"active_direction_key={self.active_direction_key}, "
            f"num_matches={num_matches}, effective_min_matches={effective_min_matches}, "
            f"spatial_coverage={spatial_coverage}, "
            f"effective_min_spatial_coverage={effective_min_spatial_coverage}, "
            f"score_mean={score_mean}, fallback_min_score_mean={self.fallback_min_score_mean}, "
            f"score_check_enabled={score_check_enabled}, "
            f"count_too_low={count_too_low}, coverage_too_low={coverage_too_low}, score_too_low={score_too_low}"
        )

        # Fast pre-check so orb->xfeat does not waste extra time probing obviously weak ORB results.
        if count_too_low or coverage_too_low or score_too_low:
            print("[ProposedMatcher._should_fallback] Fast pre-check triggered fallback before geometry probe.")
            return True

        probe = self._probe_primary_geometry(primary_result, image0_shape, image1_shape)

        probe_failed = not bool(probe["success"])
        too_few_probe_inliers = int(probe["num_inliers"]) < int(self.probe_min_inliers)
        probe_ratio_too_low = float(probe["inlier_ratio"]) < float(self.probe_min_inlier_ratio)
        probe_coverage_too_low = float(probe["inlier_coverage"]) < float(self.probe_min_inlier_coverage)
        corners_bad = not bool(probe["corners_reasonable"])

        print(
            "[ProposedMatcher._should_fallback] "
            f"probe={probe}, "
            f"probe_min_inliers={self.probe_min_inliers}, "
            f"probe_min_inlier_ratio={self.probe_min_inlier_ratio}, "
            f"probe_min_inlier_coverage={self.probe_min_inlier_coverage}"
        )
        print(
            "[ProposedMatcher._should_fallback] "
            f"probe_failed={probe_failed}, "
            f"too_few_probe_inliers={too_few_probe_inliers}, "
            f"probe_ratio_too_low={probe_ratio_too_low}, "
            f"probe_coverage_too_low={probe_coverage_too_low}, "
            f"corners_bad={corners_bad}"
        )

        fallback = bool(
            probe_failed
            or too_few_probe_inliers
            or probe_ratio_too_low
            or probe_coverage_too_low
            or corners_bad
        )

        print(f"[ProposedMatcher._should_fallback] fallback={fallback}")
        return fallback

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

        if self.fallback_enabled and self._should_fallback(
                primary_result,
                adaptation,
                image0.shape,
                image1.shape,
        ):
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
