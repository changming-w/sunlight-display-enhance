# instrument/tone_curve_high_weaken.py
# 阳光可见性 tone curve 增强（高亮弱化版）。
# 亮度统计与主入口结构与 tone_curve.py 对齐；
# 区别：Y > 200 的高亮像素使用较弱的增强幅度，避免过曝。
#
# 与 sunlight_curve 的 high_protect 区别：
#   high_protect: 高亮区把结果拉回原始值 x（完全保护，增强量为 0）
#   high_weaken:  高亮区把增强量 (y - x) 乘以衰减因子（保留部分增强）
#
# 用法：
#   from instrument.tone_curve_high_weaken import enhance_image_high_weaken
#   out = enhance_image_high_weaken(img, p=0.45, s=1.65, high_weaken=0.5)

import numpy as np

from common.utils.image_utils import (
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


# 自适应 curve_kwargs 中包含 high_protect/high_start/high_end，
# 高亮逻辑由本模块的 high_weaken 接管，这三个键直接忽略。
_IGNORED_CURVE_KEYS = ("high_protect", "high_start", "high_end")


def _filter_curve_kwargs(curve_kwargs):
    if not curve_kwargs:
        return {}
    return {k: v for k, v in curve_kwargs.items() if k not in _IGNORED_CURVE_KEYS}


# ============================================================
# 高亮弱化曲线
# ============================================================

def sunlight_curve_high_weaken(
    x,
    p=0.45,
    s=1.65,

    # 高亮削弱：从交点 p 开始平滑过渡，保证单调性
    high_weaken=0.5,                # 高亮区保留的增强比例，1.0=不削弱，0.0=完全保护
    high_weaken_start=None,         # 削弱起始（默认=p，从交点开始）
    high_weaken_end=1.0,            # 削弱完成（默认=1.0，到端点）

    s_low=None,
    brightness_boost=0.0,
    brightness_start=0.025,
    brightness_end=0.16,
    contrast_boost=1.0,
    contrast_pivot=0.42,
    shadow_protect=0.0,
    shadow_end=0.14,

    # 吸收自适应 curve_kwargs 中的 high_protect/high_start/high_end
    **_ignored_kwargs,
):
    """
    阳光可见性增强曲线（高亮弱化版）。

    设计原则：
        - 从交点 p 开始平滑过渡到完整削弱，而非窄带突变
        - 保证整条曲线单调递增

    流程：
        1. 常规 sunlight_curve 完整增强 -> y_full；
        2. 对 x > p 的区域，把增强量 (y_full - x)
           按 smoothstep 衰减为原来的 high_weaken 倍：
               y = x + (y_full - x) * (1 - (1 - high_weaken) * w)
           其中 w = smoothstep(high_weaken_start, high_weaken_end, x)。
        3. 端点保持 y(0) = 0, y(1) = 1。
    """
    x = np.asarray(x, dtype=np.float32)

    # 默认从交点 p 开始削弱
    if high_weaken_start is None:
        high_weaken_start = p

    curve_kwargs = dict(
        s_low=s_low,
        brightness_boost=brightness_boost,
        brightness_start=brightness_start,
        brightness_end=brightness_end,
        contrast_boost=contrast_boost,
        contrast_pivot=contrast_pivot,
        high_protect=0.0,        # 高亮保护交给本函数的削弱逻辑
        shadow_protect=shadow_protect,
        shadow_end=shadow_end,
    )

    y_full = sunlight_curve(x, p=p, s=s, **curve_kwargs)

    high_weaken = float(np.clip(high_weaken, 0.0, 1.0))
    if high_weaken < 1.0:
        w = smoothstep(high_weaken_start, high_weaken_end, x)
        gain_scale = 1.0 - (1.0 - high_weaken) * w  # 1.0 -> high_weaken
        y = x + (y_full - x) * gain_scale
    else:
        y = y_full

    y = np.where(x <= 0.0, 0.0, y)
    y = np.where(x >= 1.0, 1.0, y)

    return y.astype(np.float32)


def build_lut_high_weaken(dtype, p, s, high_weaken_kwargs=None, curve_kwargs=None):
    """为 uint8 / uint16 图像预计算 LUT，避免逐像素重复求值。"""
    if high_weaken_kwargs is None:
        high_weaken_kwargs = {}
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
    y = sunlight_curve_high_weaken(
        x, p=p, s=s, **high_weaken_kwargs, **curve_kwargs
    )

    lut = np.round(y * max_value)
    lut = np.clip(lut, 0, max_value).astype(lut_dtype)

    return lut


# ============================================================
# 应用方式与主入口
# ============================================================

def apply_curve_y_only_high_weaken(
    img, p, s, standard="bt709",
    high_weaken_kwargs=None, curve_kwargs=None, lux=None,
):
    """
    只调节 Y 通道，保持 Cb/Cr 不变（可选色度补偿）。
    Y > 200/255 区域使用较弱的增强幅度。

    输入为 OpenCV BGR。

    lux: 如果指定，则使用对应 lux 的色度补偿参数；否则使用默认值。
    """
    if high_weaken_kwargs is None:
        high_weaken_kwargs = {}
    curve_kwargs = _filter_curve_kwargs(curve_kwargs)

    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    Y, Cb, Cr = bgr_to_ycbcr_float(img_float, standard=standard)

    # uint8/uint16 直接查 LUT，float 逐像素求值
    lut = build_lut_high_weaken(dtype, p, s, high_weaken_kwargs, curve_kwargs)
    if lut is not None:
        Y_new = lut[(Y * max_value).astype(np.int64)] / max_value
        Y_new = Y_new.astype(np.float32)
    else:
        Y_new = sunlight_curve_high_weaken(
            Y, p=p, s=s, **high_weaken_kwargs, **curve_kwargs
        )

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


def enhance_image_high_weaken(
    img,
    p=0.45,
    s=1.65,
    high_weaken=0.5,
    high_weaken_start=200 / 255.0,
    high_weaken_end=240 / 255.0,
    standard="bt709",
    curve_kwargs=None,
    lux=None,
):
    """
    Stage1 阳光可见性 tone curve 增强（高亮弱化版）主入口。

    参数：
        img: OpenCV BGR (uint8/uint16) 或 BGRA
        p: 曲线锚点
        s: 曲线强度
        high_weaken: 高亮区(>200)保留的增强比例
            1.0 = 与原版 sunlight_curve 完全一致（不削弱）
            0.5 = 高亮区增强量减半（推荐）
            0.0 = 高亮区完全不增强（等同 high_protect=1）
        high_weaken_start: 削弱起始亮度（默认 200/255）
        high_weaken_end: 削弱完成亮度（默认 240/255）
    """
    if img is None:
        return None

    high_weaken_kwargs = {
        "high_weaken": high_weaken,
        "high_weaken_start": high_weaken_start,
        "high_weaken_end": high_weaken_end,
    }

    if img.ndim == 2:
        lut = build_lut_high_weaken(
            img.dtype, p, s, high_weaken_kwargs, _filter_curve_kwargs(curve_kwargs)
        )
        if lut is not None:
            return lut[img]

        img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
        out = sunlight_curve_high_weaken(
            img_float, p=p, s=s, **high_weaken_kwargs,
            **_filter_curve_kwargs(curve_kwargs)
        )
        return out.astype(img.dtype)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    bgr, alpha = split_bgr_alpha(img)

    bgr_out = apply_curve_y_only_high_weaken(
        bgr,
        p,
        s,
        standard=standard,
        high_weaken_kwargs=high_weaken_kwargs,
        curve_kwargs=curve_kwargs,
        lux=lux,
    )

    return merge_bgr_alpha(bgr_out, alpha)
