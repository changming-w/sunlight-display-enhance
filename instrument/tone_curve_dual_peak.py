# instrument/tone_curve_dual_peak.py
# 双峰直方图自适应增强：低亮峰、高亮峰各用一条 S 曲线拉伸峰内对比度，
# 锚点（与 y=x 的交点）分别落在两个峰中心，两峰之间围绕谷底平滑过渡。

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
from instrument.tone_curve import enhance_image as _tone_curve_enhance
from common.algorithm.chroma_comp import (
    compensate_chroma,
    is_chroma_compensation_enabled,
)
from instrument.config import (
    get_chroma_compensation_by_lux,
    get_dual_peak_params_by_lux,
    get_tone_params_by_lux,
)


# 双峰曲线内部单调保护网格密度
_GRID_N = 4097


# ============================================================
# 双峰检测
# ============================================================

def _gaussian_kernel_1d(sigma):
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_histogram_nearest(hist, sigma):
    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    padded = np.pad(hist.astype(np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _otsu_threshold(hist):
    """返回最佳分割 bin 索引（Otsu 类间方差最大）。"""
    hist = hist.astype(np.float64)
    total = hist.sum()
    n = len(hist)
    if total <= 0:
        return n // 2

    idx = np.arange(n, dtype=np.float64)
    sum_all = float(np.dot(idx, hist))

    w_b = np.cumsum(hist)
    sum_b = np.cumsum(idx * hist)

    valid = (w_b > 0) & (w_b < total)
    if not np.any(valid):
        return n // 2

    w_b_v = w_b[valid]
    w_f_v = total - w_b_v
    m_b = sum_b[valid] / w_b_v
    m_f = (sum_all - sum_b[valid]) / w_f_v
    var_between = w_b_v * w_f_v * (m_b - m_f) ** 2

    t_candidates = np.nonzero(valid)[0]
    return int(t_candidates[np.argmax(var_between)])


def _side_percentile_center(Y_side, q):
    """一侧像素集合的交点：直接取第 q 百分位。"""
    if Y_side.size == 0:
        return None
    return float(np.percentile(Y_side, q))


def find_dual_peaks(
    Y,
    max_samples=1000000,
    bins=256,
    smooth_sigma=3.0,
    min_peak_sep=0.25,
    min_peak_mass=0.15,
    min_valley_ratio=0.25,
):
    # 阈值与板端 board_dual_peak_detect.h 保持一致（严苛版，尽可能不判双峰）：
    #   min_peak_sep  0.20 -> 0.25（实测通过最小 0.310 / 回退最大 0.264）
    #   min_peak_mass 0.10 -> 0.15（实测通过最小 0.189 / 回退最大 0.134）
    #   min_valley_ratio 0.35 -> 0.25（实测通过最大 0.135 / 回退最小 0.237）
    #   MAX_ASYM      0.60 -> 0.50（实测通过最大 0.311 / 回退 07(1.11) 67(1.02)）
    #   代价：05（暗侧 mass 0.108）从双峰变为回退 S 曲线
    """
    检测 Y 直方图的双峰结构。

    流程：
        1. 平滑直方图（高斯核）；
        2. Otsu 找谷底分割，把直方图分成左右两块；
        3. 每一块取指定百分位作为交点：暗块 p80、亮块 p20；
        4. 判定是否双峰：峰间距、两侧质量、谷底深度。

    返回 dict：
        is_bimodal / p_dark / valley / p_bright / mass_dark / mass_bright 等
    """
    Y = np.clip(np.asarray(Y, dtype=np.float32), 0.0, 1.0).reshape(-1)

    if Y.size > max_samples:
        step = max(1, Y.size // max_samples)
        Y = Y[::step]

    hist, edges = np.histogram(Y, bins=bins, range=(0.0, 1.0))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    hist_s = _smooth_histogram_nearest(hist, smooth_sigma)

    total = hist_s.sum()
    if total <= 0:
        return {"is_bimodal": False, "reason": "empty histogram"}

    # Otsu 找谷
    t = _otsu_threshold(hist_s)
    valley = float(bin_centers[t])

    # 按谷底把像素分成左右两块：暗块取 p80、亮块取 p10 作为交点（锚点向中间收）
    Y_dark = Y[Y <= valley]
    Y_bright = Y[Y > valley]

    mass_dark = float(Y_dark.size / Y.size)
    mass_bright = float(Y_bright.size / Y.size)

    if Y_dark.size == 0 or Y_bright.size == 0:
        return {
            "is_bimodal": False,
            "reason": "one side empty",
            "valley": valley,
            "mass_dark": mass_dark,
            "mass_bright": mass_bright,
        }

    p_dark = _side_percentile_center(Y_dark, 80.0)
    p_bright = _side_percentile_center(Y_bright, 10.0)

    # 谷底深度判定用平滑直方图的两侧峰高
    dark_hist = hist_s[: t + 1]
    bright_hist = hist_s[t + 1:]
    peak_dark_h = float(dark_hist.max())
    peak_bright_h = float(bright_hist.max())
    valley_h = float(hist_s[t])

    sep = p_bright - p_dark
    min_peak_h = min(peak_dark_h, peak_bright_h)

    is_bimodal = (
        sep >= min_peak_sep
        and mass_dark >= min_peak_mass
        and mass_bright >= min_peak_mass
        and (min_peak_h <= 0 or valley_h <= min_valley_ratio * min_peak_h)
    )

    # 防止峰中心贴边
    p_dark = float(np.clip(p_dark, 0.005, valley - 0.01))
    p_bright = float(np.clip(p_bright, valley + 0.01, 0.995))

    return {
        "is_bimodal": bool(is_bimodal),
        "p_dark": p_dark,
        "valley": valley,
        "p_bright": p_bright,
        "mass_dark": mass_dark,
        "mass_bright": mass_bright,
        "peak_dark_h": peak_dark_h,
        "peak_bright_h": peak_bright_h,
        "valley_h": valley_h,
        "sep": sep,
    }


# ============================================================
# 谷底 V 强度调制（保守方向：大峰保守、小峰微进）
# ============================================================


def adapt_strength_by_valley(
    valley,
    s_dark,
    s_bright,
    dead_lo=0.42,
    dead_hi=0.58,
    span=0.12,
    weaken=0.35,
    boost=0.10,
):
    """
    谷底位置 V 调制双峰拉伸强度（保守方向）。

    V 偏右（暗块像素多、整图偏暗）→ 暗峰覆盖面大：
        s_dark 削弱（weaken），s_bright 微进（boost）；
    V 偏左（亮块主导、整图偏亮）→ 亮峰覆盖面大：
        s_bright 削弱，s_dark 微进；
    [dead_lo, dead_hi] 为死区，普通双峰图不调制。

    与 tone_curve v2 的 brightness score 非对称调制同构；
    板端 LUT bank (v3) 中折叠进 v 索引轴，零额外成本。
    """
    v = float(np.clip(valley, 0.0, 1.0))
    if v > dead_hi:
        w = min((v - dead_hi) / span, 1.0)
        s_dark = max(s_dark * (1.0 - weaken * w), 1.0)
        s_bright = s_bright * (1.0 + boost * w)
    elif v < dead_lo:
        w = min((dead_lo - v) / span, 1.0)
        s_dark = s_dark * (1.0 + boost * w)
        s_bright = max(s_bright * (1.0 - weaken * w), 1.0)
    return s_dark, s_bright


# ============================================================
# 双 S 曲线
# ============================================================

def _dual_peak_control_points(
    p_dark,
    valley,
    p_bright,
    s_dark,
    s_bright,
    stretch_width,
    mid_max_loss,
):
    """
    构造 PCHIP 控制点（W 型）。

    每个峰附近 3 个控制点：压暗点 → 交点(峰中心, 在 y=x 上) → 提亮点；
    谷底 1 个控制点落在 y=x 上；两端 (0,0) / (1,1)。

    偏离量 e 与峰带宽 w 的关系：e = (k-1)*w，k = max(1, s)
    为峰中心附近的目标斜率，量级与原 S 曲线的 s 一致。

    中亮区斜率损失约束：峰侧过渡带长 L 内回落 e，
    平均斜率 ≈ 1 - e/L，因此限制 e <= mid_max_loss * L，
    保证中亮区斜率 >= 1 - mid_max_loss（默认 0.3）。
    """
    # --- 暗峰 ---
    w_d = max(min(stretch_width, 0.35 * (valley - p_dark)), 1e-3)
    k_d = max(1.0, float(s_dark))
    e_d = (k_d - 1.0) * w_d
    L_d = max(valley - (p_dark + w_d), 1e-3)
    e_d = min(e_d, mid_max_loss * L_d)

    # --- 亮峰 ---
    w_b = max(min(stretch_width, 0.35 * (p_bright - valley)), 1e-3)
    k_b = max(1.0, float(s_bright))
    e_b = (k_b - 1.0) * w_b
    L_b = max((p_bright - w_b) - valley, 1e-3)
    e_b = min(e_b, mid_max_loss * L_b)

    pts = [(0.0, 0.0)]

    # 暗峰暗侧：压暗
    x1 = max(p_dark - w_d, 0.002)
    y1 = max(x1 - e_d, 0.0005)
    if x1 > 0.001 and y1 > 0.0002:
        pts.append((x1, y1))

    # 暗峰交点
    pts.append((p_dark, p_dark))

    # 暗峰亮侧：提亮
    x2 = min(p_dark + w_d, valley - 0.01)
    y2 = min(x2 + e_d, valley - 0.002)
    pts.append((x2, max(y2, p_dark + 0.002)))

    # 暗峰过渡段：从 (x2, x2+e_d) 线性回落到 (valley, valley)。
    # 钉住线性路径点，PCHIP 在共线区间导数 = 区间斜率 = 1 - e_d/L >= 1 - mid_max_loss。
    # 路径点加密到 5 个，缩短与峰拉伸段的交界区间，抑制端点导数拉高引起的内部下凹。
    for t in (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0):
        xt = x2 + t * (valley - x2)
        yt = min(xt + e_d * (1.0 - t), valley - 0.002)
        pts.append((xt, max(yt, y2 + 0.002)))

    # 谷底锚点（identity 上）
    pts.append((valley, valley))

    # 亮峰暗侧：压暗
    x3 = max(p_bright - w_b, valley + 0.01)
    y3 = max(x3 - e_b, valley + 0.002)

    # 亮峰过渡段：从 (valley, valley) 线性压暗到 (x3, x3-e_b)。
    # 同样加密到 5 个路径点，抑制交界区间下凹。
    for t in (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0):
        xt = valley + t * (x3 - valley)
        yt = max(xt - e_b * t, valley + 0.002)
        pts.append((xt, min(yt, y3 - 0.002)))

    pts.append((x3, min(y3, p_bright - 0.002)))

    # 亮峰交点
    pts.append((p_bright, p_bright))

    # 亮峰亮侧：提亮
    x4 = min(p_bright + w_b, 0.998)
    y4 = min(x4 + e_b, 0.999)
    pts.append((x4, max(y4, p_bright + 0.002)))

    pts.append((1.0, 1.0))

    # 保底：x 严格递增、y 严格递增（PCHIP 单调的前提）
    cleaned = [pts[0]]
    for xk, yk in pts[1:]:
        xk_prev, yk_prev = cleaned[-1]
        xk = max(xk, xk_prev + 1e-4)
        yk = max(yk, yk_prev + 1e-4)
        cleaned.append((xk, yk))
    return cleaned


def dual_peak_curve(
    x,
    p_dark,
    valley,
    p_bright,
    s_dark=2.0,
    s_bright=2.0,
    stretch_width=0.12,
    mid_max_loss=0.3,
):
    """
    双峰拉伸曲线（PCHIP 单调插值）。

    构造：
        - 每个峰附近用 3 个控制点形成局部拉伸
          （暗侧压暗 → 峰中心交点 → 亮侧提亮），
          峰中心附近斜率 ≈ max(1, s)；
        - 谷底控制点落在 y=x 上，中亮过渡带平均斜率
          ≈ 1 - e/L，受 mid_max_loss 约束（默认损失 <= 30%）；
        - PCHIP 保形保单调，控制点 y 严格递增 → 曲线单调。

    峰内效果（W 型偏离 y=x）：
        - x < p_dark            : y < x  压暗（暗峰的暗侧）
        - p_dark < x < valley   : y > x  提亮（暗峰的亮侧），靠近谷底回落
        - valley < x < p_bright : y < x  压暗（亮峰的暗侧），靠近谷底回落
        - x > p_bright          : y > x  提亮（亮峰的亮侧）

    参数：
        stretch_width : 峰附近拉伸带半宽（0~0.2），越大峰内拉伸范围越宽
        mid_max_loss : 中亮区斜率损失预算（0~1），0.3 = 斜率不低于 0.7
    """
    from scipy.interpolate import PchipInterpolator

    x = np.asarray(x, dtype=np.float32)

    pts = _dual_peak_control_points(
        p_dark=p_dark,
        valley=valley,
        p_bright=p_bright,
        s_dark=s_dark,
        s_bright=s_bright,
        stretch_width=stretch_width,
        mid_max_loss=mid_max_loss,
    )
    xs = np.asarray([p[0] for p in pts], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts], dtype=np.float64)

    interp = PchipInterpolator(xs, ys)

    grid = np.linspace(0.0, 1.0, _GRID_N, dtype=np.float64)
    y_grid = interp(grid)

    # 单调保护（PCHIP 理论上已单调，数值保险）
    y_grid = np.maximum.accumulate(y_grid)

    y = np.interp(x, grid, y_grid).astype(np.float32)
    y = np.where(x <= 0.0, 0.0, y)
    y = np.where(x >= 1.0, 1.0, y)

    return y


def build_lut_dual_peak(
    dtype,
    p_dark,
    valley,
    p_bright,
    s_dark=2.0,
    s_bright=2.0,
    stretch_width=0.12,
    mid_max_loss=0.3,
):
    """为 uint8 / uint16 图像预计算 LUT，避免逐像素重复求值。"""
    dtype = np.dtype(dtype)

    if dtype == np.uint8:
        n = 256
    elif dtype == np.uint16:
        n = 65536
    else:
        return None

    x = np.arange(n, dtype=np.float64) / (n - 1)
    y = dual_peak_curve(
        x,
        p_dark=p_dark,
        valley=valley,
        p_bright=p_bright,
        s_dark=s_dark,
        s_bright=s_bright,
        stretch_width=stretch_width,
        mid_max_loss=mid_max_loss,
    )

    lut = np.clip(np.rint(y * (n - 1)), 0, n - 1).astype(dtype)
    return lut


# ============================================================
# Y-only 应用（含色度补偿）
# ============================================================

def _apply_y_curve(bgr, curve_fn, standard="bt709", lux=None):
    """通用 Y-only 曲线应用：曲线作用于 Y，可选色度补偿，Cb/Cr 保持。"""
    dtype = bgr.dtype
    max_value = get_dtype_max(bgr)

    img_float = np.clip(bgr.astype(np.float32) / max_value, 0.0, 1.0)
    Y, Cb, Cr = bgr_to_ycbcr_float(img_float, standard=standard)

    Y_new = curve_fn(Y).astype(np.float32)

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


def apply_curve_y_only_dual_peak(
    img,
    peaks,
    s_dark=2.0,
    s_bright=2.0,
    stretch_width=0.12,
    mid_max_loss=0.3,
    standard="bt709",
    lux=None,
):
    """
    双峰拉伸曲线 Y-only 应用。peaks 为 find_dual_peaks 的返回 dict。

    uint8/uint16 走 LUT 加速，其余逐像素求值。
    输入为 OpenCV BGR。
    """
    lut = build_lut_dual_peak(
        img.dtype,
        p_dark=peaks["p_dark"],
        valley=peaks["valley"],
        p_bright=peaks["p_bright"],
        s_dark=s_dark,
        s_bright=s_bright,
        stretch_width=stretch_width,
        mid_max_loss=mid_max_loss,
    )

    if lut is not None:
        def curve_fn(Y):
            max_value = get_dtype_max(img)
            return lut[(Y * max_value).astype(np.int64)].astype(np.float32) / max_value
    else:
        def curve_fn(Y):
            return dual_peak_curve(
                Y,
                p_dark=peaks["p_dark"],
                valley=peaks["valley"],
                p_bright=peaks["p_bright"],
                s_dark=s_dark,
                s_bright=s_bright,
                stretch_width=stretch_width,
                mid_max_loss=mid_max_loss,
            )

    return _apply_y_curve(img, curve_fn, standard=standard, lux=lux)


# ============================================================
# 主入口
# ============================================================

def enhance_image_dual_peak(
    img,
    s_dark=None,
    s_bright=None,
    stretch_width=None,
    mid_max_loss=None,
    standard="bt709",
    lux=None,

    # 双峰检测参数
    detect_params=None,

    # 谷底 V 强度调制（保守方向：大峰保守），False 关闭
    valley_adapt=True,

    # 非双峰回退参数（走 tone_curve.enhance_image 完整算法）
    fallback_p=None,
    fallback_s=None,
    fallback_p_method="iqr",

    return_info=False,
):
    """
    Stage1 双峰自适应增强主入口。

    参数解析优先级（与原曲线管线一致）：
        1. 显式传入的 s_dark / s_bright / stretch_width / mid_max_loss；
        2. 未传且 lux 非空：从 DUAL_PEAK_LUX_PRESETS 三档配置取值
           （5000 / 45000 / 100000，中间 lux 线性平滑过渡）；
        3. 都没有：内置默认 (2.0, 2.0, 0.12, 0.3)。

    回退参数 fallback_p / fallback_s 同理：
        未传且 lux 非空时从 LUX_PRESETS (get_tone_params_by_lux) 取。

    双峰图：
        - 低亮/高亮两个峰各用一条局部拉伸（W 型），
          交点分别在两峰中心；
        - 谷底锚定在 y=x 上，中亮过渡带斜率损失
          受 mid_max_loss 约束。

    非双峰图：
        - 回退到 tone_curve.enhance_image 完整算法
          （内容自适应 p/s + YCbCr 色度补偿）。

    支持 2D 灰度、3D BGR、3D BGRA，uint8/uint16/float。
    """
    if img is None:
        return None

    # --- lux 三档配置解析（显式传参优先）---
    if lux is not None:
        dp = get_dual_peak_params_by_lux(lux)
        if s_dark is None:
            s_dark = dp["s_dark"]
        if s_bright is None:
            s_bright = dp["s_bright"]
        if stretch_width is None:
            stretch_width = dp["stretch_width"]
        if mid_max_loss is None:
            mid_max_loss = dp["mid_max_loss"]
        if fallback_p is None or fallback_s is None:
            p_lux, s_lux = get_tone_params_by_lux(lux)
            if fallback_p is None:
                fallback_p = p_lux
            if fallback_s is None:
                fallback_s = s_lux

    # 内置默认
    if s_dark is None:
        s_dark = 2.0
    if s_bright is None:
        s_bright = 2.0
    if stretch_width is None:
        stretch_width = 0.12
    if mid_max_loss is None:
        mid_max_loss = 0.3
    if fallback_p is None:
        fallback_p = 0.45
    if fallback_s is None:
        fallback_s = 1.65

    # 灰度 2D：直接查 LUT
    if img.ndim == 2:
        Y = np.clip(img.astype(np.float32), 0.0, 1.0)
        peaks = find_dual_peaks(Y, **(detect_params or {}))

        if peaks.get("is_bimodal"):
            if valley_adapt:
                s_dark, s_bright = adapt_strength_by_valley(
                    peaks["valley"], s_dark, s_bright)
            lut = build_lut_dual_peak(
                img.dtype,
                p_dark=peaks["p_dark"],
                valley=peaks["valley"],
                p_bright=peaks["p_bright"],
                s_dark=s_dark,
                s_bright=s_bright,
                stretch_width=stretch_width,
                mid_max_loss=mid_max_loss,
            )
            if lut is not None:
                out = lut[img]
            else:
                out = dual_peak_curve(
                    img,
                    p_dark=peaks["p_dark"],
                    valley=peaks["valley"],
                    p_bright=peaks["p_bright"],
                    s_dark=s_dark,
                    s_bright=s_bright,
                    stretch_width=stretch_width,
                    mid_max_loss=mid_max_loss,
                ).astype(img.dtype)
        else:
            out = _tone_curve_enhance(
                img,
                p=fallback_p,
                s=fallback_s,
                mode="ycbcr",
                content_adaptive=True,
                p_method=fallback_p_method,
            )

        if return_info:
            return out, peaks
        return out

    # 3D：拆 alpha，统计 Y 找峰
    bgr, alpha = split_bgr_alpha(img)
    Y = compute_luma_float(bgr, standard=standard).astype(np.float32)

    peaks = find_dual_peaks(Y, **(detect_params or {}))

    if peaks.get("is_bimodal"):
        if valley_adapt:
            s_dark, s_bright = adapt_strength_by_valley(
                peaks["valley"], s_dark, s_bright)
        bgr_out = apply_curve_y_only_dual_peak(
            bgr,
            peaks,
            s_dark=s_dark,
            s_bright=s_bright,
            stretch_width=stretch_width,
            mid_max_loss=mid_max_loss,
            standard=standard,
            lux=lux,
        )
    else:
        # 回退：tone_curve.enhance_image 完整算法（内容自适应 + 色度补偿）
        bgr_out = _tone_curve_enhance(
            bgr,
            p=fallback_p,
            s=fallback_s,
            mode="ycbcr",
            content_adaptive=True,
            lux=lux,
            p_method=fallback_p_method,
        )

    out = merge_bgr_alpha(bgr_out, alpha)

    if return_info:
        return out, peaks
    return out


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    import cv2
    from pathlib import Path

    print("=" * 60)
    print("1. 单调性 + 中亮斜率验证（多组峰配置）")
    print("=" * 60)

    x = np.linspace(0, 1, 10000)
    all_ok = True

    configs = [
        (0.05, 0.50, 0.95, 2.0, 2.0, 0.3),
        (0.10, 0.40, 0.90, 3.0, 3.0, 0.3),
        (0.03, 0.55, 0.96, 4.5, 2.0, 0.3),
        (0.20, 0.60, 0.80, 1.6, 2.5, 0.3),
        (0.05, 0.50, 0.95, 2.0, 4.5, 0.3),
    ]

    for p_d, v, p_b, s_d, s_b, ml in configs:
        y = dual_peak_curve(
            x, p_dark=p_d, valley=v, p_bright=p_b, s_dark=s_d, s_bright=s_b,
            mid_max_loss=ml,
        )
        slopes = np.diff(y) / (x[1] - x[0])
        non_mono = int(np.sum(slopes < -1e-6))

        # 中亮区（谷底 ±25% 内）最小斜率，应 >= 1 - mid_max_loss
        mid_mask = (x >= v - 0.10) & (x <= v + 0.10)
        min_mid_slope = float(slopes[mid_mask[:-1]].min())

        # 验证三个 y=x 交点
        cross_d = y[np.argmin(np.abs(x - p_d))]
        cross_v = y[np.argmin(np.abs(x - v))]
        cross_b = y[np.argmin(np.abs(x - p_b))]

        slope_ok = min_mid_slope >= 1.0 - ml - 0.02
        status = "OK" if (non_mono == 0 and slope_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(
            f"  p_d={p_d:.2f} v={v:.2f} p_b={p_b:.2f} s=({s_d},{s_b}) ml={ml} "
            f"单调={non_mono == 0}  中亮min斜率={min_mid_slope:.3f} "
            f"(下限 {1.0 - ml:.2f})  "
            f"y(p_d)={cross_d:.4f} y(v)={cross_v:.4f} y(p_b)={cross_b:.4f}  {status}"
        )

    print(f"\n  总检: {'PASS' if all_ok else 'FAIL'}")

    print()
    print("=" * 60)
    print("1.5 lux 三档配置解析（DUAL_PEAK_LUX_PRESETS）")
    print("=" * 60)

    from instrument.config import get_dual_peak_params_by_lux

    for lux, desc in [
        (5000, "低档"), (45000, "中档"), (100000, "高档"),
        (25000, "中点插值 t=0.5"), (2000, "下界 clamp"), (200000, "上界 clamp"),
    ]:
        d = get_dual_peak_params_by_lux(lux)
        # 插值结果也须满足单调 + 中亮预算
        y = dual_peak_curve(
            x, p_dark=0.10, valley=0.50, p_bright=0.90,
            s_dark=d["s_dark"], s_bright=d["s_bright"],
            stretch_width=d["stretch_width"], mid_max_loss=d["mid_max_loss"],
        )
        slopes = np.diff(y) / (x[1] - x[0])
        non_mono = int(np.sum(slopes < -1e-6))
        # 中亮核心过渡带：纯路径点区间 [p_dark+w, p_bright-w]，排除峰边缘交界区间
        mid_mask = (x[:-1] >= 0.10 + d["stretch_width"] + 0.05) & \
                   (x[:-1] <= 0.90 - d["stretch_width"] - 0.05)
        min_mid_slope = float(slopes[mid_mask].min())
        ok = non_mono == 0 and min_mid_slope >= 1.0 - d["mid_max_loss"] - 0.02
        print(
            f"  lux={lux:<7} {desc:<14} s={d['s_dark']:.2f}/{d['s_bright']:.2f} "
            f"ml={d['mid_max_loss']:.3f} w={d['stretch_width']:.3f}  "
            f"单调={non_mono == 0}  中亮min斜率={min_mid_slope:.3f} "
            f"(下限 {1.0 - d['mid_max_loss']:.3f})  {'OK' if ok else 'FAIL'}"
        )

    print()
    print("=" * 60)
    print("2. 车机.png 双峰检测 + 增强")
    print("=" * 60)

    img_path = Path("车机.png")
    if not img_path.exists():
        print("车机.png 不存在，跳过")
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        Y = compute_luma_float(img, standard="bt709").astype(np.float32)

        peaks = find_dual_peaks(Y)
        print(f"  is_bimodal = {peaks['is_bimodal']}")
        print(f"  p_dark     = {peaks.get('p_dark', float('nan')):.4f}")
        print(f"  valley     = {peaks.get('valley', float('nan')):.4f}")
        print(f"  p_bright   = {peaks.get('p_bright', float('nan')):.4f}")
        print(f"  mass_dark  = {peaks.get('mass_dark', float('nan')):.3f}")
        print(f"  mass_bright= {peaks.get('mass_bright', float('nan')):.3f}")

        out = enhance_image_dual_peak(img, s_dark=2.0, s_bright=2.0)

        out_dir = Path("output_车机")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "车机_dual_peak.png"
        cv2.imwrite(str(out_path), out)
        print(f"\n  [Save] {out_path}")

        # 分区统计
        def luma(im):
            return (
                0.2126 * im[:, :, 2].astype(float)
                + 0.7152 * im[:, :, 1].astype(float)
                + 0.0722 * im[:, :, 0].astype(float)
            )

        Y_in, Y_out = luma(img), luma(out)
        p_d, v, p_b = peaks["p_dark"] * 255, peaks["valley"] * 255, peaks["p_bright"] * 255

        regions = [
            ("暗峰暗侧   [0, p_d)", Y_in < p_d),
            ("暗峰亮侧 [p_d, v)", (Y_in >= p_d) & (Y_in < v)),
            ("亮峰暗侧 [v, p_b)", (Y_in >= v) & (Y_in < p_b)),
            ("亮峰亮侧 [p_b, 255]", Y_in >= p_b),
        ]

        print("\n  分区增强量（+提亮 / -压暗）：")
        for name, mask in regions:
            if mask.sum() > 0:
                delta = np.mean(Y_out[mask] - Y_in[mask])
                expect = "提亮" if "亮侧" in name else "压暗"
                print(
                    f"    {name}: {delta:+.1f}  (期望{expect})"
                )
