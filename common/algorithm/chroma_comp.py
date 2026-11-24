# chroma_compensation.py
#
# 色度补偿模块：在 Y-only tone mapping 后，对 Cb/Cr 做等比例缩放，
# 部分恢复因亮度变化而视觉丢失的饱和度。

import numpy as np

from instrument.config import CHROMA_COMPENSATION, get_chroma_compensation_by_lux


def compensate_chroma(
    Y_old,
    Y_new,
    Cb,
    Cr,
    chroma_preserve=None,
    max_chroma_gain=None,
    min_chroma_gain=None,
):
    """
    根据 Y 通道变化量，对 Cb/Cr 做色度补偿。

    原理：
        ratio = Y_new / Y_old
        chroma_gain = ratio ^ chroma_preserve
        Cb_new = Cb * chroma_gain
        Cr_new = Cr * chroma_gain

    参数：
        Y_old:  tone mapping 前的 Y 通道 (float32, [0,1])
        Y_new:  tone mapping 后的 Y 通道 (float32, [0,1])
        Cb:     色度 Cb 通道 (float32)
        Cr:     色度 Cr 通道 (float32)
        chroma_preserve:  补偿强度 (0~1)，None 时使用默认值
        max_chroma_gain:  最大色度放大倍数，None 时使用默认值
        min_chroma_gain:  最小色度缩减倍数，None 时使用默认值

    返回：
        Cb_comp, Cr_comp
    """
    if chroma_preserve is None:
        chroma_preserve = 0.20
    if max_chroma_gain is None:
        max_chroma_gain = 1.5
    if min_chroma_gain is None:
        min_chroma_gain = 1.0

    chroma_preserve = float(chroma_preserve)
    max_chroma_gain = float(max_chroma_gain)
    min_chroma_gain = float(min_chroma_gain)

    if chroma_preserve <= 0.0:
        return Cb, Cr

    eps = 1e-6

    # 计算亮度变化比率
    ratio = Y_new / np.maximum(Y_old, eps)

    # 色度增益：ratio^chroma_preserve
    # ratio > 1 (提亮) → chroma_gain > 1 → 增加饱和度补偿
    # ratio < 1 (压暗) → 不补偿，保持 gain = 1.0
    chroma_gain = np.power(ratio, chroma_preserve)

    # 只对提亮像素应用 min/max 限制；压暗像素固定 gain=1.0
    brightened = ratio > 1.0
    chroma_gain = np.where(
        brightened,
        np.clip(chroma_gain, min_chroma_gain, max_chroma_gain),
        1.0,
    )

    Cb_comp = Cb * chroma_gain
    Cr_comp = Cr * chroma_gain

    return Cb_comp.astype(np.float32), Cr_comp.astype(np.float32)

def is_chroma_compensation_enabled(lux=None):
    """
    检查色度补偿是否启用。

    如果指定 lux，检查该 lux 级别的 enabled 标志；
    否则检查是否有任何一个 lux 级别启用了色度补偿。
    """
    if lux is not None:
        params = get_chroma_compensation_by_lux(lux)
        return params.get("enabled", False)

    # 任何一个 lux 级别启用即返回 True
    for key in CHROMA_COMPENSATION:
        if CHROMA_COMPENSATION[key].get("enabled", False):
            return True
    return False
