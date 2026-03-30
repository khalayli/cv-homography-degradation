import cv2
import numpy as np


def _to_uint8(img):
    print(f"[_to_uint8] Input dtype={img.dtype}, shape={img.shape}")

    if img.dtype == np.uint8:
        return img

    img = np.clip(img, 0, 255).astype(np.uint8)
    print(f"[_to_uint8] Converted to dtype={img.dtype}")
    return img


def _validate_severity(severity):
    if not isinstance(severity, int):
        raise TypeError(f"severity must be int, got {type(severity)}")

    if severity < 0:
        raise ValueError(f"severity must be >= 0, got {severity}")

    print(f"[_validate_severity] severity={severity}")
    return severity


def apply_gaussian_blur(img, severity):
    severity = _validate_severity(severity)

    if severity == 0:
        print("[apply_gaussian_blur] severity=0, returning original image")
        return img

    sigma = 0.8 * severity
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel_size = max(3, kernel_size)

    if kernel_size % 2 == 0:
        kernel_size += 1

    print(f"[apply_gaussian_blur] severity={severity}, sigma={sigma:.2f}, kernel_size={kernel_size}")

    out = cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return out


def _build_motion_kernel(kernel_size):
    print(f"[_build_motion_kernel] kernel_size={kernel_size}")

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel.sum()

    return kernel


def apply_motion_blur(img, severity):
    severity = _validate_severity(severity)

    if severity == 0:
        print("[apply_motion_blur] severity=0, returning original image")
        return img

    kernel_size = 3 + 2 * severity

    if kernel_size % 2 == 0:
        kernel_size += 1

    print(f"[apply_motion_blur] severity={severity}, kernel_size={kernel_size}")

    kernel = _build_motion_kernel(kernel_size)
    out = cv2.filter2D(img, -1, kernel)
    return out


def apply_gaussian_noise(img, severity):
    severity = _validate_severity(severity)

    if severity == 0:
        print("[apply_gaussian_noise] severity=0, returning original image")
        return img

    std = 5.0 * severity
    print(f"[apply_gaussian_noise] severity={severity}, std={std:.2f}")

    img_f = img.astype(np.float32)
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    out = img_f + noise

    return _to_uint8(out)


def apply_jpeg_compression(img, severity):
    severity = _validate_severity(severity)

    if severity == 0:
        print("[apply_jpeg_compression] severity=0, returning original image")
        return img

    quality = max(10, 95 - severity * 15)
    print(f"[apply_jpeg_compression] severity={severity}, quality={quality}")

    img_u8 = _to_uint8(img)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(".jpg", img_u8, encode_params)

    if not ok:
        print("[apply_jpeg_compression] WARNING: JPEG encoding failed, returning original image")
        return img

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        print("[apply_jpeg_compression] WARNING: JPEG decoding failed, returning original image")
        return img

    return decoded


def apply_brightness(img, severity):
    severity = _validate_severity(severity)

    if severity == 0:
        print("[apply_brightness] severity=0, returning original image")
        return img

    delta = 20 * severity
    print(f"[apply_brightness] severity={severity}, delta={delta}")

    img_f = img.astype(np.float32)
    out = img_f + delta

    return _to_uint8(out)


def apply_contrast(img, severity):
    severity = _validate_severity(severity)

    if severity == 0:
        print("[apply_contrast] severity=0, returning original image")
        return img

    alpha = 1.0 + 0.15 * severity
    beta = 127.5 * (1.0 - alpha)

    print(f"[apply_contrast] severity={severity}, alpha={alpha:.2f}, beta={beta:.2f}")

    img_f = img.astype(np.float32)
    out = alpha * img_f + beta

    return _to_uint8(out)


def apply_corruption(img, kind, severity):
    print(f"[apply_corruption] kind={kind}, severity={severity}")

    if kind == "gaussian_blur":
        return apply_gaussian_blur(img, severity)

    elif kind == "motion_blur":
        return apply_motion_blur(img, severity)

    elif kind == "gaussian_noise":
        return apply_gaussian_noise(img, severity)

    elif kind == "jpeg_compression":
        return apply_jpeg_compression(img, severity)

    elif kind == "brightness":
        return apply_brightness(img, severity)

    elif kind == "contrast":
        return apply_contrast(img, severity)

    raise ValueError(f"Unknown corruption type: {kind}")


def apply_corruption_sequence(img, corruption_types, severity):
    print(f"[apply_corruption_sequence] corruption_types={corruption_types}, severity={severity}")

    out = img.copy()

    for kind in corruption_types:
        print(f"[apply_corruption_sequence] Applying: {kind}")
        out = apply_corruption(out, kind, severity)

    print("[apply_corruption_sequence] Done")
    return out