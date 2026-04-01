
import numpy as np
import cv2 as cv

class ORBMatcher:
    """
    Matcher wrapper for ORB.

    Purpose:
    - Initialize ORB detector/descriptor
    - Detect keypoints and compute descriptors in two images
    - Match descriptors
    - Apply match filtering
    - Return matched keypoints in a simple project-wide format

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

        self.detector = None
        self.descriptor_matcher = None

        # Safety sanity check
        if not self.cfg:
            raise ValueError("WARNING! ORB: Config is empty")


        # Parse ORB-related config
        nfeatures, scale_factor, nlevels, fast_threshold, ratio_test = (
            self.cfg["orb"]["nfeatures"],
            self.cfg["orb"]["scaleFactor"],
            self.cfg["orb"]["nlevels"],
            self.cfg["orb"]["fastThreshold"],
            self.cfg["orb"]["ratio_test"],
        )

        print("[ORBMatcher.__init__] nfeatures =", nfeatures)
        print("[ORBMatcher.__init__] scale_factor =", scale_factor)
        print("[ORBMatcher.__init__] nlevels =", nlevels)
        print("[ORBMatcher.__init__] fast_threshold =", fast_threshold)
        print("[ORBMatcher.__init__] ratio_test =", ratio_test)

        # Build the ORB detector/descriptor object
        self.detector = cv.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            fastThreshold=fast_threshold,
        )
        # Build the descriptor matcher object (cc False for ratio test)
        self.descriptor_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        # Store settings needed later in match(...)
        self.ratio_test = ratio_test

        print("[ORBMatcher.__init__] Done.")
        return

    def _preprocess_image(self, image):
        print("[ORBMatcher._preprocess_image] Preprocessing image...")
        print(f"[ORBMatcher._preprocess_image] image_shape={getattr(image, 'shape', None)}")

        # image format: (H,W,3), BGR, uint8.
        # orb format: (H,W), gray, uint8.

        # sanity type check
        if not isinstance(image, np.ndarray):
            raise ValueError("Image is not numpy array")

        # convert dtype if needed
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # BGR image
        if image.ndim == 3 and image.shape[2] == 3:
            output_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # already grayscale
        elif image.ndim == 2:
            output_image = image

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return output_image

    def _detect_and_describe(self, image):
        print("[ORBMatcher._detect_and_describe] Detecting keypoints and computing descriptors...")
        print(f"[ORBMatcher._detect_and_describe] image_shape={getattr(image, 'shape', None)}")


        # Run the ORB detector/descriptor on one image.

        # find the keypoints and descriptors with ORB
        kp, des = self.detector.detectAndCompute(image, None)
        num_keypoints = len(kp)


        # Helpful for debugging config

        if num_keypoints == 0:
            print("[ORBMatcher._detect_and_describe] No keypoints found ")

        if des is None:
            print("[ORBMatcher._detect_and_describe] Descriptors are None")

        return {
            "keypoints": kp,
            "descriptors": des,
            "num_keypoints": num_keypoints,
        }

        # Expected return:
        # {
        #     "keypoints": <list of cv2 keypoints>,
        #     "descriptors": <descriptor array or None>,
        #     "num_keypoints": <int>,
        # }


    def _match_descriptors(self, desc0, desc1):
        print("[ORBMatcher._match_descriptors] Matching descriptors...")

        # Early return on empty
        if desc0 is None or desc1 is None:
            print("[ORBMatcher._match_descriptors] One or both descriptors are None")
            return []

        if len(desc0) == 0 or len(desc1) == 0:
            print("[ORBMatcher._match_descriptors] One or both descriptors are empty")
            return []

        # Match descriptors between the two images.
        matches = self.descriptor_matcher.knnMatch(desc0, desc1, k=2)

        # Apply ratio test
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue

            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good_matches.append(m)

        # config improvement debug
        print("[ORBMatcher._match_descriptors] good_matches =", len(good_matches))

        return good_matches

        # Expected return:
        #   raw match objects from the matcher


    def _extract_matched_points(self, keypoints0, keypoints1, filtered_matches):
        print("[ORBMatcher._extract_matched_points] Extracting matched point coordinates...")

        # Convert filtered match objects + keypoints into plain Nx2 arrays.

        # Case 1 : Empty/0
        if filtered_matches is None or len(filtered_matches) == 0:
            print("[ORBMatcher._extract_matched_points] filtered_matches is None/0")
            return {
                "matched_points0": np.empty((0, 2), dtype=np.float32),
                "matched_points1": np.empty((0, 2), dtype=np.float32),
                "num_matches": 0,
                "scores": None,
            }

        # Case 2: Matches
        # matched_points0 contains (x,y) for every match in image 0
        # They are corresponding

        matched_points0 = np.array(
            [keypoints0[m.queryIdx].pt for m in filtered_matches],
            dtype=np.float32
        )
        matched_points1 = np.array(
            [keypoints1[m.trainIdx].pt for m in filtered_matches],
            dtype=np.float32
        )

        num_matches = len(filtered_matches)

        print("[ORBMatcher._extract_matched_points] num_matches =", num_matches)
        #print("[ORBMatcher._extract_matched_points] matched_points0_shape =",matched_points0.shape)
        #print("[ORBMatcher._extract_matched_points] matched_points1_shape =",matched_points1.shape)

        return {
            "matched_points0": matched_points0,
            "matched_points1": matched_points1,
            "num_matches": num_matches,
            "scores": None,
        }


        # Expected return:
        # {
        #     "matched_points0": <Nx2 float array>,
        #     "matched_points1": <Nx2 float array>,
        #     "num_matches": <int>,
        #     "scores": <optional scores or None>,
        # }

    def match(self, image0, image1):
        print("[ORBMatcher.match] Matching image pair...")
        print(f"[ORBMatcher.match] image0_shape={getattr(image0, 'shape', None)}")
        print(f"[ORBMatcher.match] image1_shape={getattr(image1, 'shape', None)}")

        # Validate input images
        if image0 is None or image1 is None:
            raise ValueError("One or both input images are None")

        if not isinstance(image0, np.ndarray) or not isinstance(image1, np.ndarray):
            raise ValueError("One or both input images are not numpy arrays")

        if image0.size == 0 or image1.size == 0:
            raise ValueError("One or both input images are empty")

        # Preprocess image0 + image1
        image0 = self._preprocess_image(image0)
        image1 = self._preprocess_image(image1)

        # Detect and describe image0 and image1
        feat0 = self._detect_and_describe(image0)
        feat1 = self._detect_and_describe(image1)

        # Match descriptors + filter
        filtered_matches = self._match_descriptors(feat0["descriptors"], feat1["descriptors"])

        # Convert filtered matches into matched point arrays
        result = self._extract_matched_points(
            feat0["keypoints"],
            feat1["keypoints"],
            filtered_matches,
        )
        # Return the final project-wide match dictionary
        return result
        # Keep output format consistent with other matchers



def build_orb_matcher(cfg=None):
    #Creates and returns a configured ORBMatcher instance.

    print("[build_orb_matcher] Building ORB matcher...")
    print(f"[build_orb_matcher] cfg={cfg}")

    matcher = ORBMatcher(cfg)

    print("[build_orb_matcher] ORB matcher built successfully")
    return matcher