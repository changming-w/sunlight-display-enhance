# metrics.py

import cv2
import numpy as np

from common.utils.image_utils import compute_luma_float


def compute_contrast_metrics(img, local_sigma=3.0):
    """
    计算图像 Y 通道对比度指标。
    """
    Y = compute_luma_float(img).astype(np.float32)
    Y = np.clip(Y, 0.0, 1.0)

    eps = 1e-8

    mean = float(np.mean(Y))
    std = float(np.std(Y))

    p5, p50, p95 = np.percentile(Y.reshape(-1), [5, 50, 95])
    p5 = float(p5)
    p50 = float(p50)
    p95 = float(p95)

    p95_p5 = p95 - p5
    michelson = p95_p5 / max(p95 + p5, eps)

    if local_sigma is not None and local_sigma > 0:
        mean_local = cv2.GaussianBlur(
            Y,
            ksize=(0, 0),
            sigmaX=float(local_sigma),
            sigmaY=float(local_sigma),
            borderType=cv2.BORDER_REFLECT101,
        )

        mean2_local = cv2.GaussianBlur(
            Y * Y,
            ksize=(0, 0),
            sigmaX=float(local_sigma),
            sigmaY=float(local_sigma),
            borderType=cv2.BORDER_REFLECT101,
        )

        local_var = np.maximum(mean2_local - mean_local * mean_local, 0.0)
        local_std = np.sqrt(local_var)
        local_std_mean = float(np.mean(local_std))
    else:
        local_std_mean = 0.0

    metrics = {
        "mean_255": mean * 255.0,
        "std_255": std * 255.0,
        "p5_255": p5 * 255.0,
        "p50_255": p50 * 255.0,
        "p95_255": p95 * 255.0,
        "p95_p5_255": p95_p5 * 255.0,
        "michelson_p95_p5": michelson,
        "local_std_mean_255": local_std_mean * 255.0,
    }

    return metrics