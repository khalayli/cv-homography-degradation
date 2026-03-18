
import numpy as np


class ORBMatcher:
    """
    Skeleton matcher wrapper for ORB.

    Purpose:
    - Initialize ORB detector/descriptor
    - Detect keypoints and compute descriptors in two images
    - Match descriptors
    - Apply match filtering
    - Return matched keypoints in a simple project-wide format

    What this file should NOT do:
    - load HPatches dataset files
    - apply corruptions
    - estimate homography
    - compute metrics
    - save results

    Expected usage in the project:
        matcher = ORBMatcher(cfg)
        result = matcher.match(image0, image1)

    Expected return format from match(...):
        {
            "matched_points0": <Nx2 array>,
            "matched_points1": <Nx2 array>,
            "num_matches": <int>,
            "scores": <optional scores or None>,
        }

    Notes:
    - Keep the output format consistent with XFeat and other matchers.
    - The rest of the pipeline should not need to know ORB-specific internals.
    """

    def __init__(self, cfg=None):
        print("[ORBMatcher.__init__] Initializing ORB matcher...")
        print(f"[ORBMatcher.__init__] cfg={cfg}")

        self.cfg = cfg or {}
        self.name = "orb"

        self.orb_cfg = {}
        self.matcher_cfg = {}
        self.detector = None
        self.descriptor_matcher = None

        # TODO:
        # 1. Parse ORB-related config values from self.cfg
        #    Example fields:
        #    - nfeatures
        #    - scaleFactor
        #    - nlevels
        #    - fastThreshold
        #    - ratio_test
        #
        # 2. Build the ORB detector/descriptor object
        #
        # 3. Build the descriptor matcher object
        #
        # 4. Store any settings needed later in match(...)

        print("[ORBMatcher.__init__] TODO: implement config parsing and ORB setup.")
        raise NotImplementedError("TODO: implement ORBMatcher.__init__")

    def _parse_config(self):
        print("[ORBMatcher._parse_config] Parsing ORB config...")

        # TODO:
        # Read config values from self.cfg.
        #
        # Suggested expected config shape:
        # {
        #     "name": "orb",
        #     "orb": {
        #         "nfeatures": 2000,
        #         "scaleFactor": 1.2,
        #         "nlevels": 8,
        #         "fastThreshold": 20,
        #         "ratio_test": 0.75,
        #     }
        # }
        #
        # Expected result:
        # - store a clean ORB config dict on self.orb_cfg
        # - store matching/filtering settings on self.matcher_cfg

        print("[ORBMatcher._parse_config] TODO: implement config parsing.")
        raise NotImplementedError("TODO: implement _parse_config")

    def _build_detector(self):
        print("[ORBMatcher._build_detector] Building ORB detector...")

        # TODO:
        # Create and return the ORB detector/descriptor object.
        #
        # Typical later work:
        # - import cv2
        # - call cv2.ORB_create(...)
        #
        # Expected return:
        #   ORB detector object

        print("[ORBMatcher._build_detector] TODO: implement ORB detector creation.")
        raise NotImplementedError("TODO: implement _build_detector")

    def _build_descriptor_matcher(self):
        print("[ORBMatcher._build_descriptor_matcher] Building descriptor matcher...")

        # TODO:
        # Create and return the descriptor matcher used for ORB descriptors.
        #
        # Typical later work:
        # - use a brute-force matcher
        # - use Hamming distance for binary descriptors
        #
        # Expected return:
        #   descriptor matcher object

        print("[ORBMatcher._build_descriptor_matcher] TODO: implement descriptor matcher creation.")
        raise NotImplementedError("TODO: implement _build_descriptor_matcher")

    def _preprocess_image(self, image):
        print("[ORBMatcher._preprocess_image] Preprocessing image...")
        print(f"[ORBMatcher._preprocess_image] image_shape={getattr(image, 'shape', None)}")

        # TODO:
        # Convert the image into the format expected by ORB.
        #
        # Typical later work:
        # - ensure numpy array
        # - ensure uint8
        # - convert color image to grayscale if needed
        #
        # Expected return:
        #   preprocessed image ready for ORB

        print("[ORBMatcher._preprocess_image] TODO: implement image preprocessing.")
        raise NotImplementedError("TODO: implement _preprocess_image")

    def _detect_and_describe(self, image):
        print("[ORBMatcher._detect_and_describe] Detecting keypoints and computing descriptors...")
        print(f"[ORBMatcher._detect_and_describe] image_shape={getattr(image, 'shape', None)}")

        # TODO:
        # Run the ORB detector/descriptor on one image.
        #
        # Expected return example:
        # {
        #     "keypoints": <list of cv2 keypoints>,
        #     "descriptors": <descriptor array or None>,
        #     "num_keypoints": <int>,
        # }
        #
        # Important:
        # - handle the case where no keypoints are found
        # - handle the case where descriptors are None

        print("[ORBMatcher._detect_and_describe] TODO: implement detect+describe.")
        raise NotImplementedError("TODO: implement _detect_and_describe")

    def _match_descriptors(self, desc0, desc1):
        print("[ORBMatcher._match_descriptors] Matching descriptors...")

        # TODO:
        # Match descriptors between the two images.
        #
        # Typical later work:
        # - use knnMatch or match
        # - keep raw matches before filtering
        #
        # Expected return:
        #   raw match objects from the matcher

        print("[ORBMatcher._match_descriptors] TODO: implement descriptor matching.")
        raise NotImplementedError("TODO: implement _match_descriptors")

    def _filter_matches(self, raw_matches):
        print("[ORBMatcher._filter_matches] Filtering raw matches...")

        # TODO:
        # Apply match filtering such as:
        # - Lowe ratio test
        # - optional distance filtering
        #
        # Expected return:
        #   filtered match list

        print("[ORBMatcher._filter_matches] TODO: implement match filtering.")
        raise NotImplementedError("TODO: implement _filter_matches")

    def _extract_matched_points(self, keypoints0, keypoints1, filtered_matches):
        print("[ORBMatcher._extract_matched_points] Extracting matched point coordinates...")

        # TODO:
        # Convert filtered match objects + keypoints into plain Nx2 arrays.
        #
        # Expected return:
        # {
        #     "matched_points0": <Nx2 float array>,
        #     "matched_points1": <Nx2 float array>,
        #     "num_matches": <int>,
        #     "scores": <optional scores or None>,
        # }

        print("[ORBMatcher._extract_matched_points] TODO: implement matched point extraction.")
        raise NotImplementedError("TODO: implement _extract_matched_points")

    def _build_empty_result(self):
        print("[ORBMatcher._build_empty_result] Building empty result...")

        result = {
            "matched_points0": np.zeros((0, 2), dtype=np.float32),
            "matched_points1": np.zeros((0, 2), dtype=np.float32),
            "num_matches": 0,
            "scores": None,
        }

        print(f"[ORBMatcher._build_empty_result] result={result}")
        return result

    def match(self, image0, image1):
        print("[ORBMatcher.match] Matching image pair...")
        print(f"[ORBMatcher.match] image0_shape={getattr(image0, 'shape', None)}")
        print(f"[ORBMatcher.match] image1_shape={getattr(image1, 'shape', None)}")

        # TODO:
        # 1. Validate input images
        # 2. Preprocess image0
        # 3. Preprocess image1
        # 4. Detect and describe image0
        # 5. Detect and describe image1
        # 6. If descriptors are missing, return _build_empty_result()
        # 7. Match descriptors
        # 8. Filter matches
        # 9. Convert filtered matches into matched point arrays
        # 10. Return the final project-wide match dictionary
        #
        # Important:
        # - Do not estimate homography here
        # - Do not compute metrics here
        # - Keep output format consistent with other matchers

        print("[ORBMatcher.match] TODO: implement ORB matching pipeline.")
        raise NotImplementedError("TODO: implement ORBMatcher.match")


def build_orb_matcher(cfg=None):
    print("[build_orb_matcher] Building ORB matcher...")
    print(f"[build_orb_matcher] cfg={cfg}")

    # TODO:
    # Optional helper constructor if your matcher factory wants to call this.

    print("[build_orb_matcher] TODO: implement helper constructor.")
    raise NotImplementedError("TODO: implement build_orb_matcher")