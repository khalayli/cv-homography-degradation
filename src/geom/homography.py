import cv2
import numpy as np


def estimate_homography(
    matched_points0,
    matched_points1,
    reproj_threshold=3.0,
    confidence=0.999,
    max_iters=5000,
):
    """
    Estimate a homography using RANSAC.

    Inputs:
    - matched_points0: Nx2 matched points from image 0
    - matched_points1: Nx2 corresponding matched points from image 1

    Returns:
        {
            "success": bool,
            "H": 3x3 homography or None,
            "inlier_mask": Nx1 uint8 mask or None,
            "num_input_matches": int,
            "num_inliers": int,
            "reproj_threshold": float,
            "confidence": float,
            "max_iters": int,
            "failure_reason": None or str,
        }
    """

    print("[estimate_homography] Starting homography estimation with RANSAC...")
    print(f"[estimate_homography] reproj_threshold={reproj_threshold}")
    print(f"[estimate_homography] confidence={confidence}")
    print(f"[estimate_homography] max_iters={max_iters}")

    validation = _validate_match_inputs(matched_points0, matched_points1)
    num_input_matches = validation["num_matches"]

    if not validation["ok"]:
        return _build_failure_result(
            num_input_matches=num_input_matches,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
            max_iters=max_iters,
            failure_reason=validation["reason"],
        )

    pts0 = _prepare_points_for_opencv(matched_points0)
    pts1 = _prepare_points_for_opencv(matched_points1)

    try:
        H, inlier_mask = cv2.findHomography(
            pts0,
            pts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(reproj_threshold),
            maxIters=int(max_iters),
            confidence=float(confidence),
        )
    except cv2.error as e:
        return _build_failure_result(
            num_input_matches=num_input_matches,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
            max_iters=max_iters,
            failure_reason=f"OpenCV error in findHomography: {e}",
        )

    if H is None:
        return _build_failure_result(
            num_input_matches=num_input_matches,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
            max_iters=max_iters,
            failure_reason="cv2.findHomography returned None",
        )

    return _build_success_result(
        H=H.astype(np.float32),
        inlier_mask=inlier_mask,
        num_input_matches=num_input_matches,
        reproj_threshold=reproj_threshold,
        confidence=confidence,
        max_iters=max_iters,
    )


def _build_failure_result(
    num_input_matches,
    reproj_threshold,
    confidence,
    max_iters,
    failure_reason,
):
    print("[_build_failure_result] Building failure result...")
    print(f"[_build_failure_result] num_input_matches={num_input_matches}")
    print(f"[_build_failure_result] failure_reason={failure_reason}")

    result = {
        "success": False,
        "H": None,
        "inlier_mask": None,
        "num_input_matches": int(num_input_matches),
        "num_inliers": 0,
        "reproj_threshold": float(reproj_threshold),
        "confidence": float(confidence),
        "max_iters": int(max_iters),
        "failure_reason": failure_reason,
    }

    print(f"[_build_failure_result] result={result}")
    return result


def _build_success_result(
    H,
    inlier_mask,
    num_input_matches,
    reproj_threshold,
    confidence,
    max_iters,
):
    print("[_build_success_result] Building success result...")
    print(f"[_build_success_result] num_input_matches={num_input_matches}")

    num_inliers = _count_inliers(inlier_mask)

    result = {
        "success": True,
        "H": H,
        "inlier_mask": inlier_mask,
        "num_input_matches": int(num_input_matches),
        "num_inliers": int(num_inliers),
        "reproj_threshold": float(reproj_threshold),
        "confidence": float(confidence),
        "max_iters": int(max_iters),
        "failure_reason": None,
    }

    print(f"[_build_success_result] result={result}")
    return result


def _validate_match_inputs(matched_points0, matched_points1):
    print("[_validate_match_inputs] Validating matched point inputs...")

    if matched_points0 is None or matched_points1 is None:
        return {
            "ok": False,
            "reason": "matched_points0 or matched_points1 is None",
            "num_matches": 0,
        }

    pts0 = np.asarray(matched_points0)
    pts1 = np.asarray(matched_points1)

    if pts0.ndim != 2 or pts1.ndim != 2:
        return {
            "ok": False,
            "reason": "matched points must be 2D arrays of shape [N, 2]",
            "num_matches": 0,
        }

    if pts0.shape[1] != 2 or pts1.shape[1] != 2:
        return {
            "ok": False,
            "reason": "matched points must have shape [N, 2]",
            "num_matches": 0,
        }

    if pts0.shape[0] != pts1.shape[0]:
        return {
            "ok": False,
            "reason": "matched_points0 and matched_points1 must have same number of rows",
            "num_matches": min(len(pts0), len(pts1)),
        }

    num_matches = int(pts0.shape[0])

    if num_matches < 4:
        return {
            "ok": False,
            "reason": "at least 4 matches are required to estimate a homography",
            "num_matches": num_matches,
        }

    return {
        "ok": True,
        "reason": None,
        "num_matches": num_matches,
    }


def _prepare_points_for_opencv(points):
    print("[_prepare_points_for_opencv] Preparing points for OpenCV...")

    pts = np.asarray(points, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape [N, 2], got {pts.shape}")

    return pts.reshape(-1, 1, 2)


def _count_inliers(inlier_mask):
    print("[_count_inliers] Counting inliers...")

    if inlier_mask is None:
        return 0

    mask = np.asarray(inlier_mask).reshape(-1)
    return int(np.sum(mask > 0))


def apply_homography_to_points(points, H):
    """
    Apply homography H to Nx2 points.
    """

    print("[apply_homography_to_points] Applying homography to points...")

    pts = np.asarray(points, dtype=np.float32)
    H = np.asarray(H, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape [N, 2], got {pts.shape}")
    if H.shape != (3, 3):
        raise ValueError(f"Expected H with shape [3, 3], got {H.shape}")

    warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return warped.astype(np.float32)


def get_image_corners(image_width, image_height):
    """
    Return 4 corners in order:
    top-left, top-right, bottom-right, bottom-left
    """

    print("[get_image_corners] Building image corner coordinates...")
    print(f"[get_image_corners] image_width={image_width}, image_height={image_height}")

    return np.array(
        [
            [0.0, 0.0],
            [float(image_width - 1), 0.0],
            [float(image_width - 1), float(image_height - 1)],
            [0.0, float(image_height - 1)],
        ],
        dtype=np.float32,
    )