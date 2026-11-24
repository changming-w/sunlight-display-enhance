# instrument/tone_curve.py
# 亮度统计、内容自适应参数调整、色调曲线应用（包含 enhance_image 主入口）。

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
# 亮度统计与内容自适应参数调整
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
        "p35": float(p35),
        "p50": float(p50),
        "p75": float(p75),
        "p85": float(p85),
        "p95": float(p95),
        "p95_p5": float(p95 - p5),
        "r_high": float(np.mean(flat > 0.75)),
        "r_very_high": float(np.mean(flat > 0.90)),
        "r_low": float(np.mean(flat < 0.25)),
        "r_very_low": float(np.mean(flat < 0.10)),
    }

    # 额外 p_adj 候选估计（用于自适应 p_method 选项）
    hist, bin_edges = np.histogram(flat, bins=128, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # p_mode: 直方图峰值 bin
    stats["p_mode"] = float(bin_centers[np.argmax(hist)])

    # p_iqr: IQR 中心，(p25+p75)/2 比 (p35+p95)/2 更左偏，让中低亮像素也落入提亮区
    stats["p_iqr"] = float((p35 + p95) / 2.0)

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


def adapt_tone_params_by_y_channel(Y, p, s, p_method="iqr"):
    """
    基于 Y (luma) 通道直方图，对 lux 预设的 p/s 做内容自适应修正。

    核心逻辑：
        1. p_adj：直接从直方图统计得到（iqr / mode / median / centroid）
        2. brightness：根据 p50 计算一个 [0, 1] 的亮暗分数（中性点 = 0.5）
        3. s_high / s_low：在预设 s 上按 brightness 方向乘不同系数
           - 亮场景（brightness > 0.5）：s_high 温和加强，s_low 大幅削弱（非对称，避免压暗）
           - 暗场景（brightness < 0.5）：s_high / s_low 一致削弱（对称，保留可见性）
    """
    stats = compute_luma_stats(Y)

    p = float(p)
    s = float(s)

    # 亮暗分数：基于 p50，归一化到 [0, 1]，中间有死区不触发调整
    # 0.0 = 极暗(p50≤0.15), 0.5 = 中性(p50 ∈ [0.30,0.55]), 1.0 = 极亮(p50≥0.80)
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

    # p_adj：根据 p_method 从直方图统计取值
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

    p_adj = float(np.clip(p_adj, 0.40, 0.87))

    # s_high / s_low：基于 brightness 对预设 s 非对称调整
    # 亮场景：s_high 温和增强、s_low 大幅削弱（避免压暗）
    # 暗场景：s_high/s_low 对称削弱（保留可见性）
    if brightness >= 0.5:
        # 亮场景：偏离中性越多，削弱越大
        b = (brightness - 0.5) * 2.0   # 归一化到 [0, 1]
        s_high = s * (1.0 + b * 0.10)
        s_low = s * (1.0 - b * 0.60)
    else:
        # 暗场景：偏离中性越多，削弱越大
        b = (0.5 - brightness) * 2.0   # 归一化到 [0, 1]
        s_high = s * (1.0 + b * 0.10)
        s_low = s * (1.0 - b * 0.40)

    s_high = float(np.clip(s_high, 1.2, 5))
    s_low = float(np.clip(s_low, 1.2, 4))

    # curve_kwargs：高光/暗部保护强度随亮暗偏移动态调整
    bright_excess = max(brightness - 0.5, 0.0) * 2.0   # [0, 1]
    dark_excess = max(0.5 - brightness, 0.0) * 2.0     # [0, 1]

    curve_kwargs = {
        "s_low": s_low,
        "brightness_boost": 0.0,
        "brightness_start": 0.10,
        "brightness_end": 0.55,
        "high_protect": 0.00 * bright_excess,
        "high_start": 1.0,
        "high_end": 1.0,
        "shadow_protect": 0.05 * dark_excess,
        "shadow_end": 0.1,
    }

    return p_adj, s_high, curve_kwargs, stats


def adapt_tone_params_by_v_channel(V, p, s, p_method="iqr"):
    """
    基于 HSV V 通道直方图，对 lux 预设的 p/s 做内容自适应修正。

    V = max(R, G, B)，分布整体偏高于 Y，阈值和系数相应调整。
    """
    stats = compute_luma_stats(V)

    p = float(p)
    s = float(s)

    # 亮暗分数：基于 p50（V 通道整体偏高于 Y，阈值右移）
    # 0.0 = 极暗(p50≤0.20), 0.5 = 中性(p50 ∈ [0.40,0.65]), 1.0 = 极亮(p50≥0.85)
    p50 = stats["p50"]
    if p50 <= 0.20:
        brightness = 0.0
    elif p50 < 0.40:
        brightness = 0.5 * (p50 - 0.20) / 0.20       # 0→0.5
    elif p50 <= 0.65:
        brightness = 0.5                               # 死区，不调整
    elif p50 < 0.85:
        brightness = 0.5 + 0.5 * (p50 - 0.65) / 0.20  # 0.5→1.0
    else:
        brightness = 1.0

    # p_adj：根据 p_method 从直方图统计取值
    if p_method == "fixed":
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

    # V 通道统计已自然右偏（V >= Y），仅做微量补偿
    p_adj = float(np.clip(p_adj, 0.40, 0.87))

    # s_high / s_low：V 通道偏高，亮场景削弱稍强（系数微调）
    if brightness >= 0.5:
        b = (brightness - 0.5) * 2.0
        s_high = s * (1.0 + b * 0.08)
        s_low = s * (1.0 - b * 0.55)
    else:
        b = (0.5 - brightness) * 2.0
        s_high = s * (1.0 + b * 0.10)
        s_low = s * (1.0 - b * 0.30)

    s_high = float(np.clip(s_high, 1.2, 5))
    s_low = float(np.clip(s_low, 1.2, 4))

    # curve_kwargs
    bright_excess = max(brightness - 0.5, 0.0) * 2.0
    dark_excess = max(0.5 - brightness, 0.0) * 2.0

    curve_kwargs = {
        "s_low": s_low,
        "brightness_boost": 0.0,
        "brightness_start": 0.10,
        "brightness_end": 0.55,
        "high_protect": 0.00 * bright_excess,
        "high_start": 1.0,
        "high_end": 1.0,
        "shadow_protect": 0.05 * dark_excess,
        "shadow_end": 0.1,
    }

    return p_adj, s_high, curve_kwargs, stats


def get_effective_tone_params(
    img,
    p,
    s,
    content_adaptive=True,
    standard="bt709",
    p_method="iqr",
    mode="ycbcr",
):
    """
    获取当前图像实际使用的 tone 参数。

    mode:
        "ycbcr" (默认): 基于 Y 通道（BT.709）统计自适应。
        "hsv": 基于 HSV 的 V 通道统计自适应（与 HSV 直接增强路径一致）。

    返回：
        p_eff, s_eff, curve_kwargs, tone_stats
    """
    if not content_adaptive:
        return float(p), float(s), {}, None

    if mode == "hsv":
        import cv2

        img_float = img.astype(np.float32) / get_dtype_max(img)
        img_float = np.clip(img_float, 0.0, 1.0)
        hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]  # V 通道
        return adapt_tone_params_by_v_channel(V, p, s, p_method=p_method)

    Y = extract_luma_for_tone_stats(img, standard=standard)
    return adapt_tone_params_by_y_channel(Y, p, s, p_method=p_method)


# ============================================================
# 色调曲线应用方式与 enhance_image 主入口
# ============================================================

def build_lut(dtype, p, s, curve_kwargs=None):
    """为 uint8 / uint16 图像预计算 sunlight_curve LUT，避免逐像素重复求值。"""
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
    """对每个通道独立套曲线（RGB/BGR 各通道分别处理，不保色）。优先用 LUT 加速。"""
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
    """基于 luma 生成增益，再回乘 BGR（保色但不保色度比例）。"""
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


def apply_curve_hsv(img_bgr_float, curve_func):
    """将 BGR 图转为 float32，在 HSV V 通道上应用 curve_func，H/S 保持不变。

    参数：
        img_bgr_float: BGR float32 [0, 1]
        curve_func: 已绑定 p/s/kwargs 的调用对象（如 partial(sunlight_curve, ...)）

    返回： BGR float32 [0, 1]
    """
    import cv2

    img = img_bgr_float.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V_old = hsv[:, :, 2]
    V_new = curve_func(V_old)
    hsv[:, :, 2] = np.clip(V_new, 0.0, 1.0)
    bgr_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(bgr_new, 0.0, 1.0).astype(np.float32)


def enhance_image(
    img,
    p,
    s,
    mode="ycbcr",
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

    mode:
        "ycbcr" (默认): 在 YCbCr 空间只调节 Y，可选色度补偿。
        "hsv": 在 HSV 空间只调节 V 通道，H/S 保持不变。
        "luma": 基于 luma 生成 gain，再回乘 BGR。
        "rgb" / "per_channel": 对每个通道独立套曲线。
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
                mode=mode,
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
    elif mode == "hsv":
        # HSV 路径：基于 V 通道统计（adapt_tone_params_by_v_channel）自适应，
        # 曲线应用到 V 通道而非 Y 通道，H/S 保持不变。
        from functools import partial

        curve_func = partial(sunlight_curve, p=p, s=s, **curve_kwargs)
        img_float = bgr.astype(np.float32) / get_dtype_max(bgr)
        img_float = np.clip(img_float, 0.0, 1.0)
        bgr_float_out = apply_curve_hsv(img_float, curve_func)
        bgr_out = restore_dtype(bgr_float_out, bgr.dtype)
    elif mode == "ycbcr":
        # YCbCr 路径（默认）：基于 Y 统计自适应，应用到 Y 通道，可选色度补偿。
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
