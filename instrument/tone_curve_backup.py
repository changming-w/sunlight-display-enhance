# instrument/algorithm/tone_curve.py
# Merged module: luminance statistics, content-adaptive parameter adjustment,
# and tone curve application methods (incl. main enhance_image API).

import numpy as np

from common.utils.image_utils import (
    compute_luma_float,
    smoothstep,
    get_dtype_max,
    restore_dtype,
    bgr_to_ycbcr_float,
    ycbcr_to_bgr_float,
    split_bgr_alpha,
    merge_bgr_alpha,
)
from common.algorithm.curve_math import sunlight_curve
from common.algorithm.chroma_comp import (
    compensate_chroma,
    is_chroma_compensation_enabled,
)
from instrument.config import get_chroma_compensation_by_lux


# ============================================================
# Luminance statistics and content-adaptive parameter adjustment
# (originally in core/luma_stats.py)
# ============================================================

def extract_luma_for_tone_stats(img, standard="bt709"):
    """
    提取用于 tone 参数自适应的 Y' luma。
    返回 float32 [0,1]。
    """
    return compute_luma_float(img, standard=standard).astype(np.float32)


def compute_luma_stats(Y, max_samples=1000000):
    """
    统计亮度分布，用于判断：
        - 高亮 UI / 白天导航
        - 暗场景
        - 低对比场景
    """
    Y = np.clip(Y.astype(np.float32), 0.0, 1.0)

    flat = Y.reshape(-1)

    if flat.size > max_samples:
        step = max(1, flat.size // max_samples)
        flat = flat[::step]

    p5, p25, p35, p50, p75, p85, p95 = np.percentile(
        flat,
        [5, 25, 35, 50, 75, 85, 95],
    )

    stats = {
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "p95_p5": float(p95 - p5),
        "r_high": float(np.mean(flat > 0.75)),
        "r_very_high": float(np.mean(flat > 0.90)),
        "r_low": float(np.mean(flat < 0.25)),
        "r_very_low": float(np.mean(flat < 0.10)),
    }

    # ============================================================
    # 额外 p_adj 候选估计（用于自适应 p_method 选项）
    # ============================================================
    hist, bin_edges = np.histogram(flat, bins=128, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # p_mode: 直方图峰值 bin
    stats["p_mode"] = float(bin_centers[np.argmax(hist)])

    # p_iqr: IQR 中心
    stats["p_iqr"] = float((p35 + p85) / 2.0)

    # p_centroid: 密集区域（>50% 峰值）的加权质心
    peak_count = hist.max()
    if peak_count > 0:
        threshold = 0.5 * peak_count
        dense_mask = hist > threshold
        if dense_mask.any():
            stats["p_centroid"] = float(
                np.average(bin_centers[dense_mask], weights=hist[dense_mask])
            )
        else:
            stats["p_centroid"] = float(p50)
    else:
        stats["p_centroid"] = float(p50)

    return stats


def adapt_tone_params_by_content(Y, p, s, p_method="iqr"):
    """
    基于图像亮度直方图，对 lux 预设的 p/s 做内容自适应修正。

    核心逻辑：
        1. p_adj：直接从直方图统计得到（iqr / mode / median / centroid）
        2. brightness：根据 p50 计算一个 [-1, +1] 的亮暗分数
        3. s_high / s_low：在预设 s 上按 brightness 方向乘不同系数
           - 亮场景（brightness > 0）：s_high 温和削弱，s_low 大幅削弱（非对称，避免压暗）
           - 暗场景（brightness < 0）：s_high / s_low 一致削弱（对称，保留可见性）
    """
    stats = compute_luma_stats(Y)

    p = float(p)
    s = float(s)

    # ============================================================
    # 亮暗分数：基于 p50，归一化到 [0, 1]，中间有死区不触发调整
    # 0.0 = 极暗(p50≤0.15)
    # 0.5 = 中性(p50 ∈ [0.30, 0.55])，不触发 s 调整
    # 1.0 = 极亮(p50≥0.80)
    # ============================================================
    p50 = stats["p50"]
    if p50 <= 0.15:
        brightness = 0.0
    elif p50 < 0.30:
        brightness = 0.5 * (p50 - 0.15) / 0.15       # 0→0.5
    elif p50 <= 0.55:
        brightness = 0.5                               # 死区，不调整
    elif p50 < 0.80:
        brightness = 0.5 + 0.5 * (p50 - 0.55) / 0.25  # 0.5→1.0
    else:
        brightness = 1.0

    # ============================================================
    # p_adj：根据 p_method 直接从直方图取值
    # ============================================================
    if p_method == "fixed":
        # 以预设 p 为基准，根据亮度分数微调：亮→右移，暗→左移
        p_adj = p + (brightness - 0.5) * 0.15
    elif p_method == "iqr":
        p_adj = stats["p_iqr"]
    elif p_method == "mode":
        p_adj = stats["p_mode"]
    elif p_method == "median":
        p_adj = stats["p50"]
    elif p_method == "centroid":
        p_adj = stats["p_centroid"]
    else:
        raise ValueError(f"Unknown p_method: {p_method}")

    p_adj = float(np.clip(p_adj, 0.40, 0.80))

    # ============================================================
    # s_high / s_low：在预设 s 上直接按 brightness 调整
    # brightness < 0.5 偏暗，> 0.5 偏亮，= 0.5 不调整
    # ============================================================
    if brightness >= 0.5:
        # 亮场景：偏离中性越多，削弱越大
        b = (brightness - 0.5) * 2.0   # 归一化到 [0, 1]
        s_high = s * (1.0 - b * 0.15)
        s_low = s * (1.0 - b * 0.45)
    else:
        # 暗场景：偏离中性越多，削弱越大
        b = (0.5 - brightness) * 2.0   # 归一化到 [0, 1]
        s_high = s * (1.0 - b * 0.20)
        s_low = s * (1.0 - b * 0.20)

    s_high = float(np.clip(s_high, 1.2, 3.0))
    s_low = float(np.clip(s_low, 1.2, 3.0))

    # ============================================================
    # curve_kwargs：保护区根据 brightness 方向开启
    # ============================================================
    bright_excess = max(brightness - 0.5, 0.0) * 2.0   # [0, 1]
    dark_excess = max(0.5 - brightness, 0.0) * 2.0     # [0, 1]

    curve_kwargs = {
        "s_low": s_low,
        "brightness_boost": 0.0,
        "brightness_start": 0.10,
        "brightness_end": 0.55,
        "high_protect": 0.15 * bright_excess,
        "high_start": 0.75,
        "high_end": 1.0,
        "shadow_protect": 0.25 * dark_excess,
        "shadow_end": 0.3,
    }

    return p_adj, s_high, curve_kwargs, stats


def get_effective_tone_params(
    img,
    p,
    s,
    content_adaptive=True,
    standard="bt709",
    p_method="iqr",
):
    """
    获取当前图像实际使用的 tone 参数。

    返回：
        p_eff, s_eff, curve_kwargs, tone_stats
    """
    if not content_adaptive:
        return float(p), float(s), {}, None

    Y = extract_luma_for_tone_stats(img, standard=standard)
    return adapt_tone_params_by_content(Y, p, s, p_method=p_method)


# ============================================================
# Tone curve application methods and main enhance_image API
# (originally in core/curve_apply.py)
# ============================================================

def build_lut(dtype, p, s, curve_kwargs=None):
    """
    为 uint8 / uint16 图像生成 LUT。
    """
    if curve_kwargs is None:
        curve_kwargs = {}

    dtype = np.dtype(dtype)

    if dtype == np.uint8:
        max_value = 255
        lut_dtype = np.uint8
    elif dtype == np.uint16:
        max_value = 65535
        lut_dtype = np.uint16
    else:
        return None

    x = np.linspace(0.0, 1.0, max_value + 1, dtype=np.float32)
    y = sunlight_curve(x, p=p, s=s, **curve_kwargs)

    lut = np.round(y * max_value)
    lut = np.clip(lut, 0, max_value).astype(lut_dtype)

    return lut


def apply_curve_per_channel(img, p, s, curve_kwargs=None):
    """
    对每个通道独立套曲线。
    """
    if curve_kwargs is None:
        curve_kwargs = {}

    dtype = img.dtype
    lut = build_lut(dtype, p, s, curve_kwargs=curve_kwargs)

    if lut is not None:
        return lut[img]

    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    out = sunlight_curve(img_float, p=p, s=s, **curve_kwargs)

    return out.astype(dtype)


def apply_curve_luma_preserve_color(img, p, s, curve_kwargs=None):
    """
    基于 luma 生成 gain，再回乘 BGR。
    """
    if curve_kwargs is None:
        curve_kwargs = {}

    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    B = img_float[..., 0]
    G = img_float[..., 1]
    R = img_float[..., 2]

    Y = 0.0722 * B + 0.7152 * G + 0.2126 * R
    Y_new = sunlight_curve(Y, p=p, s=s, **curve_kwargs)

    eps = 1e-6
    gain = Y_new / np.maximum(Y, eps)

    out = img_float * gain[..., None]
    out = np.clip(out, 0.0, 1.0)

    return restore_dtype(out, dtype)


def apply_curve_y_only_bgr(img, p, s, standard="bt709", curve_kwargs=None, lux=None):
    """
    只调节 Y 通道，保持 Cb/Cr 不变（可选色度补偿）。
    输入为 OpenCV BGR。

    lux: 如果指定，则使用对应 lux 的色度补偿参数；否则使用默认值。
    """
    if curve_kwargs is None:
        curve_kwargs = {}

    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    Y, Cb, Cr = bgr_to_ycbcr_float(img_float, standard=standard)
    Y_new = sunlight_curve(Y, p=p, s=s, **curve_kwargs)

    # 色度补偿：根据 Y 变化量等比例缩放 Cb/Cr
    if is_chroma_compensation_enabled(lux=lux):
        if lux is not None:
            chroma_params = get_chroma_compensation_by_lux(lux)
            Cb, Cr = compensate_chroma(
                Y, Y_new, Cb, Cr,
                chroma_preserve=chroma_params["chroma_preserve"],
                max_chroma_gain=chroma_params["max_chroma_gain"],
                min_chroma_gain=chroma_params["min_chroma_gain"],
            )
        else:
            Cb, Cr = compensate_chroma(Y, Y_new, Cb, Cr)

    out_float = ycbcr_to_bgr_float(Y_new, Cb, Cr, standard=standard, clip=True)

    return restore_dtype(out_float, dtype)


def enhance_image(
    img,
    p,
    s,
    mode="y_only",
    content_adaptive=True,
    curve_kwargs=None,
    return_info=False,
    lux=None,
    p_method="iqr",
):
    """
    Stage1 阳光可见性 tone curve 增强。

    content_adaptive=True 时：
        lux 给出的 p/s 会根据当前画面亮度直方图自动修正。
    """
    if img is None:
        return None

    tone_stats = None

    if curve_kwargs is None:
        if content_adaptive:
            p, s, curve_kwargs, tone_stats = get_effective_tone_params(
                img,
                p,
                s,
                content_adaptive=True,
                standard="bt709",
                p_method=p_method,
            )
        else:
            curve_kwargs = {}
    else:
        curve_kwargs = dict(curve_kwargs)

    if img.ndim == 2:
        out = apply_curve_per_channel(
            img,
            p,
            s,
            curve_kwargs=curve_kwargs,
        )

        if return_info:
            return out, {
                "p": float(p),
                "s": float(s),
                "curve_kwargs": curve_kwargs,
                "tone_stats": tone_stats,
            }

        return out

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    bgr, alpha = split_bgr_alpha(img)

    if mode in ["rgb", "per_channel"]:
        bgr_out = apply_curve_per_channel(
            bgr,
            p,
            s,
            curve_kwargs=curve_kwargs,
        )
    elif mode == "luma":
        bgr_out = apply_curve_luma_preserve_color(
            bgr,
            p,
            s,
            curve_kwargs=curve_kwargs,
        )
    elif mode in ["y_only", "yuv_y_only"]:
        bgr_out = apply_curve_y_only_bgr(
            bgr,
            p,
            s,
            standard="bt709",
            curve_kwargs=curve_kwargs,
            lux=lux,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    out = merge_bgr_alpha(bgr_out, alpha)

    if return_info:
        return out, {
            "p": float(p),
            "s": float(s),
            "curve_kwargs": curve_kwargs,
            "tone_stats": tone_stats,
        }

    return out
