# image_utils.py

import numpy as np


def get_dtype_max(img_or_dtype):
    """
    获取图像数据类型对应的最大值。
    uint8  -> 255
    uint16 -> 65535
    float  -> 1
    """
    dtype = img_or_dtype.dtype if hasattr(img_or_dtype, "dtype") else np.dtype(img_or_dtype)

    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)

    return 1.0


def to_float01(img):
    """
    将图像转换为 float32 [0,1]。
    对 float 输入默认认为已经是 [0,1]。
    """
    if img is None:
        raise ValueError("Input image is None.")

    max_value = get_dtype_max(img)
    out = img.astype(np.float32) / max_value
    return np.clip(out, 0.0, 1.0)


def restore_dtype(img_float, dtype):
    """
    将 [0,1] float 图像转回目标 dtype。
    """
    dtype = np.dtype(dtype)
    img_float = np.clip(img_float, 0.0, 1.0)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.round(img_float * float(info.max)).clip(info.min, info.max).astype(dtype)

    return img_float.astype(dtype)


def split_bgr_alpha(img):
    """
    将 BGR / BGRA 拆成 bgr 和 alpha。
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape: {img.shape}")

    c = img.shape[2]

    if c == 3:
        return img, None

    if c == 4:
        return img[..., :3], img[..., 3:4]

    raise ValueError(f"Unsupported channel count: {c}")


def merge_bgr_alpha(bgr, alpha):
    """
    将 bgr 和 alpha 合并。
    """
    if alpha is None:
        return bgr

    return np.concatenate([bgr, alpha], axis=-1)


def get_ycbcr_coeffs(standard="bt709"):
    standard = standard.lower()

    if standard == "bt601":
        Kr = 0.299
        Kb = 0.114
    elif standard == "bt709":
        Kr = 0.2126
        Kb = 0.0722
    else:
        raise ValueError(f"Unsupported YCbCr standard: {standard}")

    Kg = 1.0 - Kr - Kb
    return Kr, Kg, Kb


def bgr_to_ycbcr_float(img_float, standard="bt709"):
    """
    输入 BGR float [0,1]
    输出 full-range Y, Cb, Cr
    """
    Kr, Kg, Kb = get_ycbcr_coeffs(standard)

    B = img_float[..., 0]
    G = img_float[..., 1]
    R = img_float[..., 2]

    Y = Kr * R + Kg * G + Kb * B
    Cb = (B - Y) / (2.0 * (1.0 - Kb))
    Cr = (R - Y) / (2.0 * (1.0 - Kr))

    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def ycbcr_to_bgr_float(Y, Cb, Cr, standard="bt709", clip=True):
    """
    full-range YCbCr -> BGR float。
    """
    Kr, Kg, Kb = get_ycbcr_coeffs(standard)

    R = Y + 2.0 * (1.0 - Kr) * Cr
    B = Y + 2.0 * (1.0 - Kb) * Cb
    G = (Y - Kr * R - Kb * B) / Kg

    out = np.stack([B, G, R], axis=-1).astype(np.float32)

    if clip:
        out = np.clip(out, 0.0, 1.0)

    return out


def compute_luma_float(img, standard="bt709"):
    """
    计算归一化 luma，范围 [0,1]。
    输入默认是 OpenCV BGR/BGRA。
    """
    if img is None:
        raise ValueError("Input image is None.")

    max_value = get_dtype_max(img)
    # 如果是limit，归一化时8bit图像应该除以（235-16），因为Y的范围是16-235

    if img.ndim == 2:
        Y = img.astype(np.float32) / max_value
        return np.clip(Y, 0.0, 1.0)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]

        if img.shape[2] != 3:
            raise ValueError(f"Unsupported image shape for luma: {img.shape}")

        img_float = img.astype(np.float32) / max_value
        img_float = np.clip(img_float, 0.0, 1.0)

        Y, _, _ = bgr_to_ycbcr_float(img_float, standard=standard)
        return np.clip(Y, 0.0, 1.0)

    raise ValueError(f"Unsupported image shape for luma: {img.shape}")


def image_to_rgb_float_for_plot(img):
    """
    将 OpenCV 读取的 BGR/BGRA/Gray 图像转成 matplotlib 显示用 RGB float [0,1]。
    """
    if img is None:
        raise ValueError("Input image is None.")

    max_value = get_dtype_max(img)

    if img.ndim == 2:
        gray = img.astype(np.float32) / max_value
        gray = np.clip(gray, 0.0, 1.0)
        return np.stack([gray, gray, gray], axis=-1)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]

        if img.shape[2] != 3:
            raise ValueError(f"Unsupported image shape for plot: {img.shape}")

        bgr = img.astype(np.float32) / max_value
        bgr = np.clip(bgr, 0.0, 1.0)

        return bgr[..., ::-1]

    raise ValueError(f"Unsupported image shape for plot: {img.shape}")


def smoothstep(edge0, edge1, x):
    """
    平滑阈值函数。
    x 可以是标量，也可以是 ndarray。
    """
    denom = max(float(edge1) - float(edge0), 1e-6)
    t = (x - float(edge0)) / denom
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def percentile_fast(x, percentile=99.5, max_samples=1000000):
    """
    大图上计算 percentile 时做简单采样，避免太慢。
    """
    flat = np.asarray(x).reshape(-1)

    if flat.size == 0:
        return 0.0

    if flat.size > max_samples:
        step = max(1, flat.size // max_samples)
        flat = flat[::step]

    return float(np.percentile(flat, percentile))