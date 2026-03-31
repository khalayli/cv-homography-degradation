import numpy as np


class ProposedMatcher:
    """
    Skeleton for the project's proposed degradation-aware matcher.

    Purpose:
    - Wrap the proposed method in one place
    - Run a primary matcher
    - Optionally estimate degradation indicators
    - Optionally adapt matcher settings
    - Optionally suggest geometry overrides for homography fitting
    - Optionally fall back to another matcher
    - Return matches in the same format as all other matchers

    What this file should NOT do:
    - load HPatches scenes
    - apply corruptions
    - estimate homography directly
    - compute metrics
    - save CSV/JSON

    Expected usage:
        matcher = ProposedMatcher(cfg)
        result = matcher.match(image0, image1)

    Expected return format from match(...):
        {
            "matched_points0": <Nx2 array>,
            "matched_points1": <Nx2 array>,
            "num_matches": <int>,
            "scores": <optional scores or None>,
            "backend_used": <string>,
            "fallback_used": <0 or 1>,
            "geom_overrides": {
                "reproj_threshold": <optional float>,
                "confidence": <optional float>,
                "max_iters": <optional int>,
            },
        }
    """

    def __init__(self, cfg=None):
        print("[ProposedMatcher.__init__] Initializing proposed matcher...")
        print(f"[ProposedMatcher.__init__] cfg={cfg}")

        self.cfg = cfg or {}
        self.name = "proposed"

        self.proposed_cfg = {}
        self.primary_matcher = None
        self.fallback_matcher = None
        # TODO:
        # 1. Parse the method config and keep only the proposed section
        # 2. Read settings like:
        #    - adaptive_thresholds
        #    - fallback_enabled
        #    - primary_matcher
        #    - fallback_matcher
        #    - min_matches_before_fallback
        #    - degradation thresholds
        # 3. Build the primary matcher
        # 4. Optionally build the fallback matcher

        self._parse_config()

        try:
            self.primary_matcher = self._build_primary_matcher()
        except Exception as e:
            print(f"[ProposedMatcher] Primary matcher failed to init: {e}")
            self.primary_matcher = None

        try:
            self.fallback_matcher = self._build_fallback_matcher()
        except Exception as e:
            print(f"[ProposedMatcher] Fallback matcher failed to init: {e}")
            self.fallback_matcher = None


    def _parse_config(self):
        print("[ProposedMatcher._parse_config] Parsing proposed matcher config...")

        # TODO:
        # Read self.cfg and extract the proposed-method settings.
        #
        # Suggested config shape:
        # {
        #     "name": "proposed",
        #     "orb": {...},
        #     "xfeat": {...},
        #     "proposed": {
        #         "enabled": True,
        #         "adaptive_thresholds": True,
        #         "fallback_enabled": True,
        #         "primary_matcher": "xfeat",
        #         "fallback_matcher": "orb",
        #         "min_matches_before_fallback": 30,
        #     }
        # }
        self.proposed_cfg = self.cfg.get("proposed", {})

        self.primary_name = self.proposed_cfg.get("primary_matcher", "orb")
        self.fallback_name = self.proposed_cfg.get("fallback_matcher", "orb")
        self.fallback_enabled = self.proposed_cfg.get("fallback_enabled", True)
        self.min_matches = self.proposed_cfg.get("min_matches_before_fallback", 30)


    def _build_primary_matcher(self):
        print("[ProposedMatcher._build_primary_matcher] Building primary matcher...")

        # TODO:
        # Build and return the primary matcher selected in config.
        #
        # Examples:
        # - xfeat
        # - orb
        if self.primary_name == "orb":
            from .orb import ORBMatcher
            return ORBMatcher(self.cfg)

        elif self.primary_name == "xfeat":
            from .xfeat import XFeatMatcher
            return XFeatMatcher(self.cfg)

        return None


    def _build_fallback_matcher(self):
        print("[ProposedMatcher._build_fallback_matcher] Building fallback matcher...")

        # TODO:
        # Build and return the fallback matcher if enabled.
        # Otherwise return None.
        if not self.fallback_enabled:
            return None

        if self.fallback_name == "orb":
            from .orb import ORBMatcher
            return ORBMatcher(self.cfg)

        elif self.fallback_name == "xfeat":
            from .xfeat import XFeatMatcher
            return XFeatMatcher(self.cfg)

        print(f"[ProposedMatcher] Unknown fallback matcher: {self.fallback_name}")
        return None


    def _estimate_degradation_indicators(self, image0, image1):
        print("[ProposedMatcher._estimate_degradation_indicators] Estimating degradation indicators...")

        # TODO:
        # Compute cheap image-quality indicators such as:
        # - blur proxy
        # - noise proxy
        # - contrast / brightness proxy
        # - compression proxy
        #
        # Expected return example:
        # {
        #     "blur_score": ...,
        #     "noise_score": ...,
        #     "contrast_score": ...,
        #     "jpeg_score": ...,
        # }

        print("[ProposedMatcher._estimate_degradation_indicators] TODO: implement degradation indicators.")
        raise NotImplementedError("TODO: implement _estimate_degradation_indicators")

    def _adapt_matcher_settings(self, indicators):
        print("[ProposedMatcher._adapt_matcher_settings] Adapting matcher settings...")
        print(f"[ProposedMatcher._adapt_matcher_settings] indicators={indicators}")

        # TODO:
        # Decide how matching settings should change under degradation.
        #
        # Possible outputs:
        # - more keypoints
        # - looser ratio test
        # - lower score threshold
        # - choose different backend
        #
        # Expected return example:
        # {
        #     "backend": "xfeat",
        #     "min_matches_before_fallback": 30,
        #     "ratio_test": 0.8,
        #     "nfeatures": 3000,
        # }

        print("[ProposedMatcher._adapt_matcher_settings] TODO: implement matcher adaptation.")
        raise NotImplementedError("TODO: implement _adapt_matcher_settings")

    def _suggest_geom_overrides(self, indicators):
        print("[ProposedMatcher._suggest_geom_overrides] Suggesting geometry overrides...")
        print(f"[ProposedMatcher._suggest_geom_overrides] indicators={indicators}")

        # TODO:
        # Decide whether homography fitting settings should change.
        #
        # Example:
        # - increase reproj_threshold under blur/noise
        #
        # Expected return example:
        # {
        #     "reproj_threshold": 5.0,
        #     "confidence": 0.999,
        #     "max_iters": 5000,
        # }
        #
        # If no override is needed, return {}.

        print("[ProposedMatcher._suggest_geom_overrides] TODO: implement geometry overrides.")
        raise NotImplementedError("TODO: implement _suggest_geom_overrides")

    def _run_primary(self, image0, image1):
        print("[ProposedMatcher._run_primary] Running primary matcher...")

        # TODO:
        # Call the primary matcher on the image pair and return the raw result.
        if self.primary_matcher is None:
            return None

        try:
            return self.primary_matcher.match(image0, image1)
        except Exception as e:
            print(f"[ProposedMatcher] Primary matcher failed during match: {e}")
            return None

    def _run_fallback(self, image0, image1):
        print("[ProposedMatcher._run_fallback] Running fallback matcher...")

        # TODO:
        # Call the fallback matcher if enabled and return the raw result.

        print("[ProposedMatcher._run_fallback] TODO: implement fallback matching call.")
        raise NotImplementedError("TODO: implement _run_fallback")

    def _should_fallback(self, primary_result):
        print("[ProposedMatcher._should_fallback] Deciding whether to fallback...")
        print(f"[ProposedMatcher._should_fallback] primary_result={primary_result}")

        # TODO:
        # Decide whether the primary result is too weak.
        #
        # Example rules:
        # - too few matches
        # - low confidence
        # - poor coverage
        if primary_result is None:
            return True

        num_matches = primary_result.get("num_matches", 0)
        return num_matches < self.min_matches


    def _normalize_match_result(self, raw_result, backend_used, fallback_used, geom_overrides):
        print("[ProposedMatcher._normalize_match_result] Normalizing result...")
        print(f"[ProposedMatcher._normalize_match_result] backend_used={backend_used}, fallback_used={fallback_used}")
        print(f"[ProposedMatcher._normalize_match_result] geom_overrides={geom_overrides}")

        # TODO:
        # Convert raw matcher output into the shared project format.
        #
        # Expected return:
        # {
        #     "matched_points0": ...,
        #     "matched_points1": ...,
        #     "num_matches": ...,
        #     "scores": ...,
        #     "backend_used": backend_used,
        #     "fallback_used": fallback_used,
        #     "geom_overrides": geom_overrides,
        # }
        if raw_result is None:
            return self._build_empty_result(backend_used, fallback_used, geom_overrides)

        return {
            "matched_points0": raw_result.get("matched_points0", np.zeros((0, 2), dtype=np.float32)),
            "matched_points1": raw_result.get("matched_points1", np.zeros((0, 2), dtype=np.float32)),
            "num_matches": raw_result.get("num_matches", 0),
            "scores": raw_result.get("scores"),
            "backend_used": backend_used,
            "fallback_used": int(fallback_used),
            "geom_overrides": geom_overrides or {},
        }


    def _build_empty_result(self, backend_used="none", fallback_used=0, geom_overrides=None):
        print("[ProposedMatcher._build_empty_result] Building empty result...")

        if geom_overrides is None:
            geom_overrides = {}

        result = {
            "matched_points0": np.zeros((0, 2), dtype=np.float32),
            "matched_points1": np.zeros((0, 2), dtype=np.float32),
            "num_matches": 0,
            "scores": None,
            "backend_used": backend_used,
            "fallback_used": int(fallback_used),
            "geom_overrides": geom_overrides,
        }

        print(f"[ProposedMatcher._build_empty_result] result={result}")
        return result

    def match(self, image0, image1):
        print("[ProposedMatcher.match] Matching image pair with proposed method...")
        print(f"[ProposedMatcher.match] image0_shape={getattr(image0, 'shape', None)}")
        print(f"[ProposedMatcher.match] image1_shape={getattr(image1, 'shape', None)}")

        # TODO:
        # 1. Validate inputs
        # 2. Estimate degradation indicators
        # 3. Adapt matching settings
        # 4. Suggest geometry overrides
        # 5. Run primary matcher
        # 6. Decide whether fallback is needed
        # 7. Run fallback if needed
        # 8. Normalize final result
        # 9. Return final match dictionary
        #
        # Important:
        # - Do not estimate homography here
        # - Do not compute metrics here
        geom_overrides = {}

        primary_result = self._run_primary(image0, image1)

        if self.fallback_enabled and self._should_fallback(primary_result):
            fallback_result = self._run_fallback(image0, image1)

            return self._normalize_match_result(
                fallback_result,
                backend_used=self.fallback_name,
                fallback_used=1,
                geom_overrides=geom_overrides
            )

        return self._normalize_match_result(
            primary_result,
            backend_used=self.primary_name,
            fallback_used=0,
            geom_overrides=geom_overrides
        )


def build_proposed_matcher(cfg=None):
    print("[build_proposed_matcher] Building proposed matcher...")
    print(f"[build_proposed_matcher] cfg={cfg}")

    # TODO:
    # Optional helper constructor if your matcher factory wants to call this.

    print("[build_proposed_matcher] TODO: implement helper constructor.")
    raise NotImplementedError("TODO: implement build_proposed_matcher")
