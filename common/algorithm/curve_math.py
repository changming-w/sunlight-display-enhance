# core/curve_math.py
# S-curve mathematical foundations

import numpy as np

from common.utils.image_utils import smoothstep


def sigmoid(z):
    z = np.asarray(z, dtype=np.float32)
    return np.where(
        z >= 0.0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def sunlight_curve(
    x,
    p=0.45,
    s=1.65,

    # 分区 s：x<p 用 s_low，x>=p 用 s (s_high)
    s_low=None,

    # 新增：中间调亮度 boost
    brightness_boost=0.0,
    brightness_start=0.025,
    brightness_end=0.16,

    # 新增：输出对比度 boost
    contrast_boost=1.0,
    contrast_pivot=0.42,

    # 高亮保护
    high_protect=0.0,
    high_start=0.90,
    high_end=0.995,

    # 暗部保护
    shadow_protect=0.0,
    shadow_end=0.14,
):
    """
    阳光可见性增强曲线，强增强版本。

    流程：
        1. 基础 S 曲线增强；
        2. 输出对比度 boost；
        3. 中间调亮度 boost；
        4. 高亮保护，防止接近白区域爆掉；
        5. 暗部保护，避免极暗区域被压死。

    参数：
        p:
            曲线与 y=x 的交点。
            p 越低，中间调越容易被提亮。

        s:
            S 曲线强度。
            s 越大，对比度越强。

        brightness_boost:
            中间调亮度提升强度。
            推荐范围 0.05 ~ 0.25。

        contrast_boost:
            输出端额外对比度增强。
            推荐范围 1.00 ~ 1.28。
    """
    x = np.asarray(x, dtype=np.float32)

    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    s = float(s)

    # ========================================================
    # 1. 基础 S 曲线（分区增强）
    #
    # s_low 控制 x<p 区域（中低亮），s 控制 x>=p 区域（高亮）。
    # 在 x=p 处 y=p 不受 s 值影响，保证连续性。
    # ========================================================
    if s_low is not None:
        s_eff = np.where(x < p, float(s_low), s)
    else:
        s_eff = s
    z = s_eff * logit(x) + (1.0 - s_eff) * logit(p)
    y = sigmoid(z)

    # ========================================================
    # 2. 输出对比度增强
    #
    # 围绕 contrast_pivot 做轻量 contrast stretch。
    # contrast_pivot 取 0.42，可以让中高亮区域更积极地抬升。
    # ========================================================
    if contrast_boost is not None and abs(float(contrast_boost) - 1.0) > 1e-6:
        contrast_boost = float(contrast_boost)
        contrast_pivot = float(contrast_pivot)

        y = contrast_pivot + (y - contrast_pivot) * contrast_boost
        y = np.clip(y, 0.0, 1.0)

    # ========================================================
    # 3. 中间调亮度 boost
    #
    # y = y + boost * (1 - y)
    #
    # 特点：
    #   - 中间调提升明显；
    #   - 接近白的区域提升自然变小；
    #   - 配合 brightness_start/end 避免纯黑被抬成灰。
    # ========================================================
    if brightness_boost is not None and brightness_boost > 0:
        brightness_boost = float(brightness_boost)

        w_bright = smoothstep(brightness_start, brightness_end, x)
        y = y + brightness_boost * w_bright * (1.0 - y)
        y = np.clip(y, 0.0, 1.0)

    # ========================================================
    # 4. 高亮保护
    #
    # 只对接近白的区域生效，避免白底 UI / 天空 / 高光爆白。
    # 注意新版 high_start 默认 0.90，比旧版更晚介入，
    # 不会过早削弱中高亮增强。
    # ========================================================
    if high_protect is not None and high_protect > 0:
        high_protect = float(high_protect)

        w_high = smoothstep(high_start, high_end, x)
        y = y * (1.0 - high_protect * w_high) + x * (high_protect * w_high)
        y = np.clip(y, 0.0, 1.0)

    # ========================================================
    # 5. 暗部保护
    #
    # 只保护极暗区域，避免 S 曲线把黑位压死。
    # ========================================================
    if shadow_protect is not None and shadow_protect > 0:
        shadow_protect = float(shadow_protect)

        w_shadow = 1.0 - smoothstep(0.0, shadow_end, x)
        y = y * (1.0 - shadow_protect * w_shadow) + x * (shadow_protect * w_shadow)
        y = np.clip(y, 0.0, 1.0)

    y = np.where(x <= 0.0, 0.0, y)
    y = np.where(x >= 1.0, 1.0, y)

    return y.astype(np.float32)
