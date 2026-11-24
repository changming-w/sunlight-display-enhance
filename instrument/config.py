# config.py

import math

LUX_PRESETS = {
    1000: {
        "p": 0.55,
        "s": 1.00,
    },    
    5000: {
        "p": 0.50,
        "s": 1.60,
    },
    20000: {
        "p": 0.48,
        "s": 2.05,
    },
    40000: {
        "p": 0.45,
        "s": 2.50,
    },
    60000: {
        "p": 0.43,
        "s": 3.10,
    },
    100000: {
        "p": 0.40,
        "s": 4.10,
    },
}

DEFAULT_LUX_LIST = [1000, 5000, 20000, 40000, 60000, 80000]


# ============================================================
# Stage2 Sobel Edge Enhancement Presets
# ============================================================

EDGE_PRESETS = {
    5000: {
        "amount": 0.38,
        "sigma": 1.00,
        "outline_strength": 0.0,
        "edge_low": 0.05,
        "edge_high": 0.20,
        "edge_blur": 2.00,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
    },
    45000: {
        "amount": 0.70,
        "sigma": 1.50,
        "outline_strength": 0.0,
        "edge_low": 0.04,
        "edge_high": 0.15,
        "edge_blur": 2.50,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
    },
    100000: {
        "amount": 0.88,
        "sigma": 1.80,
        "outline_strength": 0.0,
        "edge_low": 0.05,
        "edge_high": 0.12,
        "edge_blur": 3.00,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
    },
}


# ============================================================
# Stage3 White Balance Drift Presets (Conservative)
# D65 -> ~D67 / D70 / ~D73 / D75
# Reduced shift to avoid excessive OOG for saturated images.
# ============================================================

D65_XY = (0.3127, 0.3290)

# 近似 daylight locus 白点（保守版）
D67_XY = (0.3090, 0.3254)   # ~6700K, very mild shift
D70_XY = (0.3053, 0.3216)   # ~7000K
D73_XY = (0.3020, 0.3180)   # ~7300K
D75_XY = (0.2990, 0.3149)   # ~7500K

WB_LUX_PRESETS = {
    5000: {
        "src_white_xy": D65_XY,
        "dst_white_xy": D67_XY,
        "name": "~D67",
    },
    45000: {
        "src_white_xy": D65_XY,
        "dst_white_xy": D73_XY,
        "name": "~D73",
    },
    100000: {
        "src_white_xy": D65_XY,
        "dst_white_xy": D75_XY,
        "name": "D75",
    },
}


def lux_to_name(lux):
    """
    将 lux 数值转成目录名，例如 5000 -> 5000lux。
    """
    lux = float(lux)

    if not math.isfinite(lux):
        raise ValueError(f"Invalid lux value: {lux}")

    if abs(lux - round(lux)) < 1e-6:
        return f"{int(round(lux))}lux"

    return f"{lux:g}lux"


def _find_exact_key(value, keys, eps=1e-6):
    value = float(value)

    for key in keys:
        if abs(value - float(key)) <= eps:
            return key

    return None


def _interp_scalar(v0, v1, t):
    return float(v0) * (1.0 - float(t)) + float(v1) * float(t)


def _interp_xy(xy0, xy1, t):
    x = _interp_scalar(xy0[0], xy1[0], t)
    y = _interp_scalar(xy0[1], xy1[1], t)
    return float(x), float(y)


def get_tone_params_by_lux(lux):
    """
    根据 lux 获取 Stage1 tone curve 参数 p 和 s。

    规则：
        1. lux 精确命中 preset 时，直接使用；
        2. lux 在 preset 区间内时，线性插值；
        3. lux 超出边界时，使用最近一档。
    """
    lux = float(lux)
    preset_lux = sorted(LUX_PRESETS.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        item = LUX_PRESETS[exact_key]
        return float(item["p"]), float(item["s"])

    if lux <= preset_lux[0]:
        item = LUX_PRESETS[preset_lux[0]]
        return float(item["p"]), float(item["s"])

    if lux >= preset_lux[-1]:
        item = LUX_PRESETS[preset_lux[-1]]
        return float(item["p"]), float(item["s"])

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)

            p0 = LUX_PRESETS[l0]["p"]
            s0 = LUX_PRESETS[l0]["s"]
            p1 = LUX_PRESETS[l1]["p"]
            s1 = LUX_PRESETS[l1]["s"]

            p = _interp_scalar(p0, p1, t)
            s = _interp_scalar(s0, s1, t)

            return p, s

    raise RuntimeError("Unexpected lux interpolation error.")


# ============================================================
# Dual-Peak W Curve Presets (双峰 W 曲线)
#
# 与 LUX_PRESETS 同样结构：三档 lux 参数，中间线性平滑过渡。
#   s_dark / s_bright : 峰中心附近目标斜率（越大拉伸越强）
#   mid_max_loss      : 中亮区斜率损失预算（0~1）
#   stretch_width     : 峰内拉伸带半宽 (0~0.2)
# ============================================================

DUAL_PEAK_LUX_PRESETS = {
    1000: {
        "s_dark": 1.80,
        "s_bright": 1.80,
        "mid_max_loss": 0.25,
        "stretch_width": 0.10,
    },
    5000: {
        "s_dark": 1.80,
        "s_bright": 1.80,
        "mid_max_loss": 0.25,
        "stretch_width": 0.10,
    },
    20000: {
        "s_dark": 3.37,
        "s_bright": 3.37,
        "mid_max_loss": 0.36,
        "stretch_width": 0.10,
    },
    40000: {
        "s_dark": 5.47,
        "s_bright": 5.47,
        "mid_max_loss": 0.51,
        "stretch_width": 0.11,
    },
    60000: {
        "s_dark": 6.50,
        "s_bright": 6.50,
        "mid_max_loss": 0.57,
        "stretch_width": 0.12,
    },
    100000: {
        "s_dark": 8.00,
        "s_bright": 8.00,
        "mid_max_loss": 0.65,
        "stretch_width": 0.14,
    },
}


def get_dual_peak_params_by_lux(lux):
    """
    根据 lux 获取双峰 W 曲线参数。

    规则（与 get_tone_params_by_lux 一致）：
        1. lux 精确命中 preset 时，直接使用；
        2. lux 在 preset 区间内时，线性插值（平滑过渡）；
        3. lux 超出边界时，使用最近一档。
    """
    lux = float(lux)
    preset_lux = sorted(DUAL_PEAK_LUX_PRESETS.keys())
    keys = ("s_dark", "s_bright", "mid_max_loss", "stretch_width")

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        item = DUAL_PEAK_LUX_PRESETS[exact_key]
        return {k: float(item[k]) for k in keys}

    if lux <= preset_lux[0]:
        item = DUAL_PEAK_LUX_PRESETS[preset_lux[0]]
        return {k: float(item[k]) for k in keys}

    if lux >= preset_lux[-1]:
        item = DUAL_PEAK_LUX_PRESETS[preset_lux[-1]]
        return {k: float(item[k]) for k in keys}

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)
            item0 = DUAL_PEAK_LUX_PRESETS[l0]
            item1 = DUAL_PEAK_LUX_PRESETS[l1]

            return {
                k: _interp_scalar(item0[k], item1[k], t) for k in keys
            }

    raise RuntimeError("Unexpected dual-peak lux interpolation error.")


# ============================================================
# Dark-Lift Curve Presets (v4 车机一暗一亮场景：0-30 压暗 / 30-128 提亮 / 128-192 渐进)
#
# 曲线： y = x - dark*phi_d(x) + bright*phi_b(x)   （dark/bright 为峰值幅度，灰阶）
#   x <= 30/255      : phi_d 平滑 bump，压暗（峰值 dark 灰阶，峰位 ~14 灰阶）
#   30/255 < x < 128/255: phi_b 提亮（升 [30,55]，平台 [55,128] 满幅，峰值 bright 灰阶）
#   128/255 <= x < 192/255: 提亮幅度 smoothstep 渐进衰减到 0
#   x >= 192/255     : 两 bump 均为 0，精确 identity（δ=0 自动保持）
#
# 幅度随 lux 变化：1000 档 0/0（恒等，与 v2 1000 档惯例一致），
# lux 越强压暗/提亮越多。幅度上限由带宽斜率预算约束（单调不折叠）：
#   压暗带 30 灰阶宽，峰值 <= 6 灰阶；提亮降沿 64 灰阶宽，峰值 <= 42 灰阶。
# ============================================================

DARK_LIFT_LUX_PRESETS = {
    1000: {
        "dark": 0.0,
        "bright": 0.0,
    },
    5000: {
        "dark": 1.5,
        "bright": 6.0,
    },
    20000: {
        "dark": 2.5,
        "bright": 11.0,
    },
    40000: {
        "dark": 3.5,
        "bright": 16.0,
    },
    60000: {
        "dark": 5.0,
        "bright": 21.0,
    },
    100000: {
        "dark": 6.0,
        "bright": 28.0,
    },
}


def get_dark_lift_params_by_lux(lux):
    """
    根据 lux 获取 v4 压暗/提亮曲线参数（dark/bright 为峰值幅度，灰阶）。

    规则（与 get_tone_params_by_lux 一致）：
        1. lux 精确命中 preset 时，直接使用；
        2. lux 在 preset 区间内时，线性插值（平滑过渡）；
        3. lux 超出边界时，使用最近一档。
    """
    lux = float(lux)
    preset_lux = sorted(DARK_LIFT_LUX_PRESETS.keys())
    keys = ("dark", "bright")

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        item = DARK_LIFT_LUX_PRESETS[exact_key]
        return {k: float(item[k]) for k in keys}

    if lux <= preset_lux[0]:
        item = DARK_LIFT_LUX_PRESETS[preset_lux[0]]
        return {k: float(item[k]) for k in keys}

    if lux >= preset_lux[-1]:
        item = DARK_LIFT_LUX_PRESETS[preset_lux[-1]]
        return {k: float(item[k]) for k in keys}

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)
            item0 = DARK_LIFT_LUX_PRESETS[l0]
            item1 = DARK_LIFT_LUX_PRESETS[l1]

            return {
                k: _interp_scalar(item0[k], item1[k], t) for k in keys
            }

    raise RuntimeError("Unexpected dark-lift lux interpolation error.")


def get_wb_params_by_lux(lux):
    """
    根据 lux 获取 Stage3 白平衡漂移参数（保守版）。

    默认：
        5000   lux -> D65 到 ~D67 (0.3090, 0.3254)
        45000  lux -> D65 到 ~D73 (0.3020, 0.3180)
        100000 lux -> D65 到 D75  (0.2990, 0.3149)

    如果 lux 不在 preset 中，则对 dst_white_xy 做线性插值。
    超出边界时使用最近一档。
    """
    lux = float(lux)
    preset_lux = sorted(WB_LUX_PRESETS.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        item = WB_LUX_PRESETS[exact_key]
        return {
            "src_white_xy": item["src_white_xy"],
            "dst_white_xy": item["dst_white_xy"],
            "name": item["name"],
        }

    if lux <= preset_lux[0]:
        item = WB_LUX_PRESETS[preset_lux[0]]
        return {
            "src_white_xy": item["src_white_xy"],
            "dst_white_xy": item["dst_white_xy"],
            "name": item["name"],
        }

    if lux >= preset_lux[-1]:
        item = WB_LUX_PRESETS[preset_lux[-1]]
        return {
            "src_white_xy": item["src_white_xy"],
            "dst_white_xy": item["dst_white_xy"],
            "name": item["name"],
        }

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)

            item0 = WB_LUX_PRESETS[l0]
            item1 = WB_LUX_PRESETS[l1]

            dst_white_xy = _interp_xy(
                item0["dst_white_xy"],
                item1["dst_white_xy"],
                t,
            )

            return {
                "src_white_xy": D65_XY,
                "dst_white_xy": dst_white_xy,
                "name": f"{item0['name']}-{item1['name']}-interp",
            }

    raise RuntimeError("Unexpected WB lux interpolation error.")


# ============================================================
# Chroma Compensation (色度补偿) — per-lux
#
# 当 Y 通道 tone mapping 后亮度变化时，Cb/Cr 保持不变会导致
# 饱和度视觉上降低（提亮时）或升高（压暗时）。
# 色度补偿通过 ratio^chroma_preserve 来部分恢复原始饱和度感知。
#
# chroma_preserve = 0.0 → 不补偿（Cb/Cr 不变）
# chroma_preserve = 1.0 → 完全等比例补偿
#
# 高 lux 时 S 曲线增强更强 (s: 1.50→2.60)，因此需要更强的色度补偿。
# ============================================================

CHROMA_COMPENSATION = {
    5000: {
        "enabled": True,
        "chroma_preserve": 0.55,    # s=1.50, mild enhancement
        "max_chroma_gain": 2.0,
        "min_chroma_gain": 1.10,
    },
    45000: {
        "enabled": True,
        "chroma_preserve": 0.80,    # s=2.30, moderate enhancement
        "max_chroma_gain": 2.5,
        "min_chroma_gain": 1.20,
    },
    100000: {
        "enabled": True,
        "chroma_preserve": 0.95,    # s=2.60, strong enhancement
        "max_chroma_gain": 3.0,
        "min_chroma_gain": 1.30,
    },
}


def get_chroma_compensation_by_lux(lux):
    """
    根据 lux 获取色度补偿参数。

    规则：
        1. lux 精确命中 preset 时，直接使用；
        2. lux 在 preset 区间内时，线性插值；
        3. lux 超出边界时，使用最近一档。
    """
    lux = float(lux)
    preset_lux = sorted(CHROMA_COMPENSATION.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        item = CHROMA_COMPENSATION[exact_key]
        return {
            "enabled": item["enabled"],
            "chroma_preserve": float(item["chroma_preserve"]),
            "max_chroma_gain": float(item["max_chroma_gain"]),
            "min_chroma_gain": float(item["min_chroma_gain"]),
        }

    if lux <= preset_lux[0]:
        item = CHROMA_COMPENSATION[preset_lux[0]]
        return {
            "enabled": item["enabled"],
            "chroma_preserve": float(item["chroma_preserve"]),
            "max_chroma_gain": float(item["max_chroma_gain"]),
            "min_chroma_gain": float(item["min_chroma_gain"]),
        }

    if lux >= preset_lux[-1]:
        item = CHROMA_COMPENSATION[preset_lux[-1]]
        return {
            "enabled": item["enabled"],
            "chroma_preserve": float(item["chroma_preserve"]),
            "max_chroma_gain": float(item["max_chroma_gain"]),
            "min_chroma_gain": float(item["min_chroma_gain"]),
        }

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)

            item0 = CHROMA_COMPENSATION[l0]
            item1 = CHROMA_COMPENSATION[l1]

            return {
                "enabled": item0["enabled"] or item1["enabled"],
                "chroma_preserve": _interp_scalar(
                    item0["chroma_preserve"], item1["chroma_preserve"], t
                ),
                "max_chroma_gain": _interp_scalar(
                    item0["max_chroma_gain"], item1["max_chroma_gain"], t
                ),
                "min_chroma_gain": _interp_scalar(
                    item0["min_chroma_gain"], item1["min_chroma_gain"], t
                ),
            }

    raise RuntimeError("Unexpected chroma compensation lux interpolation error.")


# ============================================================
# V4: Content-Aware Edge Enhancement Presets
# 像素级内容感知边缘增强 - 结构张量 + 不对称增强 + 选择性暗描边
# ============================================================

CONTENT_AWARE_EDGE_PRESETS = {
    5000: {
        "sigma": 5.0,
        "lambda_bright": 1.50,
        "lambda_dark": 0.60,
        "outline_dark": 0.30,
        "edge_low": 0.02,
        "edge_high": 0.12,
        "coherence_low": 0.30,
        "coherence_high": 0.70,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "sigma_st": 2.0,
        "textlike_win": 11,
        "norm_percentile": 99.5,
    },
    45000: {
        "sigma": 7.0,
        "lambda_bright": 2.20,
        "lambda_dark": 0.80,
        "outline_dark": 0.50,
        "edge_low": 0.015,
        "edge_high": 0.08,
        "coherence_low": 0.30,
        "coherence_high": 0.70,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "sigma_st": 2.0,
        "textlike_win": 11,
        "norm_percentile": 99.5,
    },
    100000: {
        "sigma": 9.0,
        "lambda_bright": 2.80,
        "lambda_dark": 1.00,
        "outline_dark": 0.65,
        "edge_low": 0.01,
        "edge_high": 0.06,
        "coherence_low": 0.30,
        "coherence_high": 0.70,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "sigma_st": 2.0,
        "textlike_win": 11,
        "norm_percentile": 99.5,
    },
}