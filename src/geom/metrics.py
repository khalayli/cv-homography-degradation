import cv2
import numpy as np


def get_value(obj, key, default=None):
    print(f"[get_value] key={key}")

    if isinstance(obj, dict):
        value = obj.get(key, default)
        print(f"[get_value] Got dict value={value}")
        return value

    value = getattr(obj, key, default)
    print(f"[get_value] Got attribute value={value}")
    return value


def image_corners(width, height):
    print(f"[image_corners] width={width}, height={height}")

    corners = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )

    print(f"[image_corners] corners_shape={corners.shape}")
    return corners


def warp_points(points, H):
    print(f"[warp_points] points_shape={points.shape}")

    points = np.asarray(points, dtype=np.float32)
    H = np.asarray(H, dtype=np.float64)

    warped = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        H,
    ).reshape(-1, 2)

    print(f"[warp_points] warped_shape={warped.shape}")
    return warped.astype(np.float32)


def corner_transfer_errors(H_pred, H_gt, width, height):
    print(
        f"[corner_transfer_errors] width={width}, height={height}, "
        f"H_pred_is_none={H_pred is None}, H_gt_is_none={H_gt is None}"
    )

    corners = image_corners(width, height)
    pred = warp_points(corners, H_pred)
    gt = warp_points(corners, H_gt)

    errors = np.linalg.norm(pred - gt, axis=1)
    print(f"[corner_transfer_errors] errors={errors.tolist()}")
    return errors.astype(np.float32)


def mean_corner_error(H_pred, H_gt, width, height):
    print("[mean_corner_error] Computing mean corner error")

    errors = corner_transfer_errors(
        H_pred=H_pred,
        H_gt=H_gt,
        width=width,
        height=height,
    )

    value = float(np.mean(errors))
    print(f"[mean_corner_error] value={value}")
    return value


def success_at_thresholds(error_value, thresholds):
    print(f"[success_at_thresholds] error_value={error_value}, thresholds={thresholds}")

    result = {}
    for threshold in thresholds:
        key = f"success@{threshold}"
        result[key] = int(error_value <= float(threshold))

    print(f"[success_at_thresholds] result={result}")
    return result


def build_pair_metrics(
    scene_name,
    split,
    pair_index,
    matcher_name,
    corruption_name,
    severity,
    runtime_s,
    image_width,
    image_height,
    H_gt,
    H_result,
    thresholds,
):
    print(
        "[build_pair_metrics] "
        f"scene_name={scene_name}, split={split}, pair_index={pair_index}, "
        f"matcher_name={matcher_name}, corruption_name={corruption_name}, "
        f"severity={severity}, runtime_s={runtime_s}"
    )

    num_matches = get_value(H_result, "num_input_matches", 0)
    num_inliers = get_value(H_result, "num_inliers", 0)
    reproj_rmse = get_value(H_result, "reproj_rmse", None)
    estimation_success = get_value(H_result, "success", False)
    H_pred = get_value(H_result, "H", None)

    row = {
        "scene_name": scene_name,
        "split": split,
        "pair_index": pair_index,
        "matcher": matcher_name,
        "corruption": corruption_name,
        "severity": severity,
        "runtime_s": float(runtime_s),
        "num_matches": int(num_matches),
        "num_inliers": int(num_inliers),
        "reproj_rmse": float(reproj_rmse) if reproj_rmse is not None else "",
        "homography_success": int(estimation_success),
    }

    if H_pred is None:
        print("[build_pair_metrics] No predicted homography available, marking corner error as inf")
        row["mean_corner_error"] = float("inf")
        row.update(success_at_thresholds(float("inf"), thresholds))
        print(f"[build_pair_metrics] row={row}")
        return row

    error_value = mean_corner_error(
        H_pred=H_pred,
        H_gt=H_gt,
        width=image_width,
        height=image_height,
    )

    row["mean_corner_error"] = float(error_value)
    row.update(success_at_thresholds(error_value, thresholds))

    print(f"[build_pair_metrics] row={row}")
    return row