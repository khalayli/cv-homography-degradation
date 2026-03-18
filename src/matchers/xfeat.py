# src/matchers/xfeat.py

import numpy as np


class XFeatMatcher:
    """
    Skeleton matcher wrapper for XFeat.

    Purpose:
    - Load or initialize the XFeat model
    - Accept two images
    - Run feature extraction + matching
    - Return matched keypoints in a simple project-wide format

    What this file should NOT do:
    - load HPatches dataset files
    - apply corruptions
    - estimate homography
    - compute metrics
    - save results

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
    - Keep the output format consistent with ORB and other matchers.
    - The rest of the pipeline should not need to know XFeat-specific internals.
    """

    def __init__(self, cfg=None):
        print("[XFeatMatcher.__init__] Initializing XFeat matcher...")
        print(f"[XFeatMatcher.__init__] cfg={cfg}")

        self.cfg = cfg or {}
        self.name = "xfeat"
        self.model = None

        # TODO:
        # 1. Parse useful config values from self.cfg
        #    Example possibilities:
        #    - top_k
        #    - device
        #    - weights path
        #    - max_keypoints
        #    - score threshold
        #
        # 2. Check whether required libraries are installed
        #
        # 3. Load the XFeat model or prepare any needed objects
        #
        # 4. Store runtime options on self for later use in match(...)

        print("[XFeatMatcher.__init__] TODO: implement config parsing and model loading.")
        raise NotImplementedError("TODO: implement XFeatMatcher.__init__")

    def _check_dependencies(self):
        print("[XFeatMatcher._check_dependencies] Checking XFeat dependencies...")

        # TODO:
        # Check whether the packages needed for XFeat are installed.
        #
        # Examples:
        # - torch
        # - xfeat package or local implementation
        #
        # Decide how you want failures to behave:
        # - raise ImportError with a clear message
        # - or print a message and fail later

        print("[XFeatMatcher._check_dependencies] TODO: implement dependency checks.")
        raise NotImplementedError("TODO: implement _check_dependencies")

    def _load_model(self):
        print("[XFeatMatcher._load_model] Loading XFeat model...")

        # TODO:
        # 1. Import the XFeat model code
        # 2. Initialize the model
        # 3. Move it to the configured device if needed
        # 4. Put it in eval mode if needed
        # 5. Return the loaded model object
        #
        # Expected return:
        #   loaded XFeat model object

        print("[XFeatMatcher._load_model] TODO: implement model loading.")
        raise NotImplementedError("TODO: implement _load_model")

    def _preprocess_image(self, image):
        print("[XFeatMatcher._preprocess_image] Preprocessing image...")
        print(f"[XFeatMatcher._preprocess_image] image_shape={getattr(image, 'shape', None)}")

        # TODO:
        # Convert the input image into the format expected by XFeat.
        #
        # Possible work here:
        # - ensure numpy array
        # - check dtype
        # - convert BGR/RGB if needed
        # - normalize pixel values if needed
        # - convert to tensor if needed
        # - add batch dimension if needed
        #
        # Important:
        # - Do not resize unless the project decides to do so consistently.
        # - Keep this logic isolated here so match(...) stays clean.
        #
        # Expected return:
        #   preprocessed image in XFeat-ready format

        print("[XFeatMatcher._preprocess_image] TODO: implement image preprocessing.")
        raise NotImplementedError("TODO: implement _preprocess_image")

    def _run_model(self, image0, image1):
        print("[XFeatMatcher._run_model] Running XFeat model on image pair...")

        # TODO:
        # Run the actual XFeat inference / matching pipeline here.
        #
        # Depending on the XFeat API you use, this may:
        # - extract features separately from image0 and image1
        # - then match descriptors
        # OR
        # - call one high-level API that returns matches directly
        #
        # Expected raw output:
        #   whatever the XFeat library returns before you convert it to your project format

        print("[XFeatMatcher._run_model] TODO: implement XFeat inference/matching.")
        raise NotImplementedError("TODO: implement _run_model")

    def _parse_model_output(self, raw_output):
        print("[XFeatMatcher._parse_model_output] Parsing raw XFeat output...")

        # TODO:
        # Convert raw XFeat outputs into plain matched point arrays.
        #
        # Required final pieces:
        # - matched_points0: Nx2 array
        # - matched_points1: Nx2 array
        #
        # Optional:
        # - scores
        #
        # Expected return:
        #   {
        #       "matched_points0": ...,
        #       "matched_points1": ...,
        #       "num_matches": ...,
        #       "scores": ...,
        #   }

        print("[XFeatMatcher._parse_model_output] TODO: implement output parsing.")
        raise NotImplementedError("TODO: implement _parse_model_output")

    def _build_empty_result(self):
        print("[XFeatMatcher._build_empty_result] Building empty result...")

        # This helper keeps the return format consistent even when matching fails
        # or finds no correspondences.

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

        # TODO:
        # 1. Validate input images
        # 2. Preprocess image0
        # 3. Preprocess image1
        # 4. Run XFeat on the pair
        # 5. Parse the raw output
        # 6. If no matches are found, return _build_empty_result()
        # 7. Return the final project-wide match dictionary
        #
        # Important:
        # - Keep the output format consistent with ORB and future matchers.
        # - Do not estimate homography here.
        # - Do not compute metrics here.

        print("[XFeatMatcher.match] TODO: implement XFeat matching pipeline.")
        raise NotImplementedError("TODO: implement XFeatMatcher.match")


def build_xfeat_matcher(cfg=None):
    print("[build_xfeat_matcher] Building XFeat matcher...")
    print(f"[build_xfeat_matcher] cfg={cfg}")

    # TODO:
    # Optional helper if you want a simple constructor function used by build_matcher(...)

    print("[build_xfeat_matcher] TODO: implement helper constructor.")
    raise NotImplementedError("TODO: implement build_xfeat_matcher")
