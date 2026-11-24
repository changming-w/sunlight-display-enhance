# entertainment_curve.py
#
# 娱乐屏增强曲线模块
# 曲线公式：L_out = (log(1 + k*L_in) / log(1 + k)) * (1 - a) + a
#
# 适用场景：娱乐屏（视频、游戏）在阳光下的可读性增强。
# 与仪表屏 S 曲线的区别：
#   - S 曲线：对比度增强，保留暗部细节，高亮推白
#   - 对数曲线：暗部大幅提亮，高亮温和压缩，整体动态范围压缩

from pathlib import Path

import numpy as np

from entertainment.config import (
    ENTERTAINMENT_LUX_PRESETS,
    ENTERTAINMENT_DEFAULT_LUX_LIST,
    ENTERTAINMENT_CHROMA_COMPENSATION,
)
from common.utils.image_utils import (
    get_dtype_max,
    restore_dtype,
    bgr_to_ycbcr_float,
    ycbcr_to_bgr_float,
    split_bgr_alpha,
    merge_bgr_alpha,
)
from common.algorithm.chroma_comp import compensate_chroma


# ============================================================
# 1. 核心曲线
# ============================================================

def entertainment_curve(x, k=10.0, a=0.05):
    """
    娱乐屏对数增强曲线。

    公式：
        L_out = (log(1 + k * L_in) / log(1 + k)) * (1 - a) + a

    参数：
        x:  输入亮度 [0, 1]
        k:  对数映射强度。k 越大，暗部提升越明显，高亮压缩越强。
            推荐范围: 3 ~ 50
        a:  环境光偏移 (ambient offset)。
            表示阳光下屏幕黑位的最低可见亮度。
            推荐范围: 0.0 ~ 0.20

    返回：
        L_out [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    k = float(max(k, 1e-6))
    a = float(np.clip(a, 0.0, 1.0))

    # 对数映射：将 [0,1] 映射到 [0,1]，暗部提升，高亮压缩
    log_mapped = np.log1p(k * x) / np.log1p(k)

    # 加入环境光偏移：整体抬升黑位
    y = log_mapped * (1.0 - a) + a

    # 保证端点
    y = np.where(x <= 0.0, a, y)
    y = np.where(x >= 1.0, 1.0, y)

    return y.astype(np.float32)


# ============================================================
# 2. Lux 参数查询（线性插值）
# ============================================================

def _find_exact_key(value, keys, eps=1e-6):
    value = float(value)
    for key in keys:
        if abs(value - float(key)) <= eps:
            return key
    return None


def _interp_scalar(v0, v1, t):
    return float(v0) * (1.0 - float(t)) + float(v1) * float(t)


def get_entertainment_params_by_lux(lux):
    """
    根据 lux 获取娱乐屏曲线参数 k 和 a。
    支持线性插值和边界截断。
    """
    lux = float(lux)
    presets = ENTERTAINMENT_LUX_PRESETS
    preset_lux = sorted(presets.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        item = presets[exact_key]
        return float(item["k"]), float(item["a"])

    if lux <= preset_lux[0]:
        item = presets[preset_lux[0]]
        return float(item["k"]), float(item["a"])

    if lux >= preset_lux[-1]:
        item = presets[preset_lux[-1]]
        return float(item["k"]), float(item["a"])

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)

            k = _interp_scalar(presets[l0]["k"], presets[l1]["k"], t)
            a = _interp_scalar(presets[l0]["a"], presets[l1]["a"], t)

            return k, a

    raise RuntimeError("Unexpected entertainment lux interpolation error.")


# ============================================================
# 3. Lux 色度补偿参数查询（线性插值）
# ============================================================

def get_chroma_compensation_by_lux(lux):
    """
    根据 lux 获取娱乐屏色度补偿参数。
    支持精确匹配、线性插值和边界截断，与 get_entertainment_params_by_lux 逻辑一致。
    """
    lux = float(lux)
    presets = ENTERTAINMENT_CHROMA_COMPENSATION
    preset_lux = sorted(presets.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        return dict(presets[exact_key])

    if lux <= preset_lux[0]:
        return dict(presets[preset_lux[0]])

    if lux >= preset_lux[-1]:
        return dict(presets[preset_lux[-1]])

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)
            p0, p1 = presets[l0], presets[l1]
            return {
                "enabled": p0.get("enabled", True) or p1.get("enabled", True),
                "chroma_preserve": _interp_scalar(p0["chroma_preserve"], p1["chroma_preserve"], t),
                "max_chroma_gain": _interp_scalar(p0["max_chroma_gain"], p1["max_chroma_gain"], t),
                "min_chroma_gain": _interp_scalar(p0["min_chroma_gain"], p1["min_chroma_gain"], t),
            }

    raise RuntimeError("Unexpected chroma compensation lux interpolation error.")


# ============================================================
# 4. 图像增强接口
# ============================================================

def apply_entertainment_curve_y_only(img, k, a, lux=None, standard="bt709"):
    """
    只调节 Y 通道（对数曲线），保持 Cb/Cr 不变（可选色度补偿）。
    输入为 OpenCV BGR。

    lux: 光照强度，用于查询对应 lux 的色度补偿参数。
         若为 None，则使用最低 lux 档位的参数作为后备。
    """
    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    Y, Cb, Cr = bgr_to_ycbcr_float(img_float, standard=standard)
    Y_new = entertainment_curve(Y, k=k, a=a)

    # 色度补偿（根据 lux 查询对应参数）
    chroma_cfg = get_chroma_compensation_by_lux(lux) if lux is not None else \
        dict(ENTERTAINMENT_CHROMA_COMPENSATION[sorted(ENTERTAINMENT_CHROMA_COMPENSATION.keys())[0]])

    if chroma_cfg.get("enabled", False):
        Cb, Cr = compensate_chroma(
            Y, Y_new, Cb, Cr,
            chroma_preserve=chroma_cfg.get("chroma_preserve"),
            max_chroma_gain=chroma_cfg.get("max_chroma_gain"),
            min_chroma_gain=chroma_cfg.get("min_chroma_gain"),
        )

    out_float = ycbcr_to_bgr_float(Y_new, Cb, Cr, standard=standard, clip=True)

    return restore_dtype(out_float, dtype)


def enhance_image_entertainment(
    img,
    k,
    a,
    lux=None,
    return_info=False,
):
    """
    娱乐屏 Stage1 增强入口。

    lux: 光照强度，用于查询对应 lux 的色度补偿参数。
    """
    if img is None:
        return None

    if img.ndim == 2:
        # 灰度图直接套曲线
        dtype = img.dtype
        max_value = get_dtype_max(img)
        img_float = img.astype(np.float32) / max_value
        img_float = np.clip(img_float, 0.0, 1.0)
        out_float = entertainment_curve(img_float, k=k, a=a)
        out = restore_dtype(out_float, dtype)

        if return_info:
            return out, {"k": float(k), "a": float(a)}
        return out

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    bgr, alpha = split_bgr_alpha(img)
    bgr_out = apply_entertainment_curve_y_only(bgr, k=k, a=a, lux=lux, standard="bt709")
    out = merge_bgr_alpha(bgr_out, alpha)

    if return_info:
        return out, {"k": float(k), "a": float(a)}
    return out


# ============================================================
# 5. 曲线绘图
# ============================================================

def save_entertainment_curve_plot(save_path, lux_list=None):
    """
    保存娱乐屏曲线图。
    """
    import matplotlib.pyplot as plt
    from config import lux_to_name

    if lux_list is None:
        lux_list = ENTERTAINMENT_DEFAULT_LUX_LIST

    x = np.linspace(0.0, 1.0, 1001)

    plt.figure(figsize=(8, 6), dpi=160)

    plt.plot(x, x, "k--", linewidth=1.4, alpha=0.7, label="y = x")

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]

    for i, lux in enumerate(lux_list):
        k, a = get_entertainment_params_by_lux(lux)
        y = entertainment_curve(x, k=k, a=a)

        color = colors[i % len(colors)]
        plt.plot(
            x, y,
            linewidth=2.4,
            color=color,
            label=f"{lux_to_name(lux)}, k={k:.1f}, a={a:.3f}",
        )

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.45)
    plt.xlabel("Input luminance L_in")
    plt.ylabel("Output luminance L_out")
    plt.title("Entertainment Screen Log Curve")
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path)
    plt.close()

    print(f"Entertainment curve plot saved to: {save_path}")
