"""
Content-Aware Edge Enhancement
像素级内容感知边缘增强 - 结构张量 + 局部对比度 + 不对称亮暗分离
"""
import cv2
import numpy as np

from common.utils.image_utils import (
    get_dtype_max,
    restore_dtype,
    bgr_to_ycbcr_float,
    ycbcr_to_bgr_float,
    compute_luma_float,
    smoothstep,
    percentile_fast,
)

from instrument.config import CONTENT_AWARE_EDGE_PRESETS


# ============================================================
# 1. 结构张量方向一致性
# ============================================================

def compute_coherence(Y, sigma_st=2.0):
    """
    结构张量 coherence = (λ1 - λ2)^2 / (λ1 + λ2 + eps)^2
    方向一致性高 → 文字/线条；低 → 噪声/纹理
    返回 [0, 1] 的 coherence map
    """
    Gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    Gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)

    Jxx = cv2.GaussianBlur(Gx * Gx, (0, 0), sigma_st, borderType=cv2.BORDER_REFLECT101)
    Jxy = cv2.GaussianBlur(Gx * Gy, (0, 0), sigma_st, borderType=cv2.BORDER_REFLECT101)
    Jyy = cv2.GaussianBlur(Gy * Gy, (0, 0), sigma_st, borderType=cv2.BORDER_REFLECT101)

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
    lambda1 = (trace + disc) * 0.5
    lambda2 = (trace - disc) * 0.5

    denom = lambda1 + lambda2 + 1e-7
    coherence = ((lambda1 - lambda2) / denom) ** 2

    return np.clip(coherence.astype(np.float32), 0.0, 1.0)


# ============================================================
# 2. 文字/图标局部特征权重
# ============================================================

def compute_textlike_weight(Y, grad_norm, coherence, win=11,
                            contrast_low=0.08, contrast_high=0.25,
                            density_low=0.05, density_high=0.15,
                            density_suppress_low=0.55, density_suppress_high=0.75,
                            coherence_low=0.3, coherence_high=0.7,
                            edge_thresh=0.15):
    """
    启发式文字/图标检测：
    - local_contrast: 局部 std / (mean + eps)
    - edge_density: 局部窗口内强边缘占比 (中等=文字，极高=纹理)
    - coherence: 方向一致性 (高=线条结构)
    """
    kernel = np.ones((win, win), np.float32) / float(win * win)

    Y_mean = cv2.filter2D(Y, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)
    Y_sq_mean = cv2.filter2D(Y * Y, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)
    local_std = np.sqrt(np.maximum(Y_sq_mean - Y_mean * Y_mean, 0.0))
    local_contrast = local_std / (Y_mean + 0.01)

    # edge_density
    edge_binary = (grad_norm > edge_thresh).astype(np.float32)
    edge_density = cv2.filter2D(edge_binary, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)

    # 组合
    w_contrast = smoothstep(contrast_low, contrast_high, local_contrast)
    w_density = (smoothstep(density_low, density_high, edge_density) *
                 (1.0 - smoothstep(density_suppress_low, density_suppress_high, edge_density)))
    w_coherence = smoothstep(coherence_low, coherence_high, coherence)

    W_textlike = w_contrast * w_density * w_coherence
    return np.clip(W_textlike.astype(np.float32), 0.0, 1.0)


# ============================================================
# 3. Y 通道内容感知增强核心
# ============================================================

def content_aware_enhance_y(
    Y,
    sigma=6.0,
    lambda_bright=2.0,
    lambda_dark=1.0,
    outline_dark=0.5,
    edge_low=0.02,
    edge_high=0.10,
    coherence_low=0.3,
    coherence_high=0.7,
    luma_low=0.02,
    luma_high=0.10,
    sigma_st=2.0,
    textlike_win=11,
    norm_percentile=99.5,
):
    """
    Y 通道内容感知边缘增强核心逻辑。

    流水线：
    1. Sobel 梯度 → M_edge
    2. 结构张量 → coherence → W_structure
    3. 局部统计 → W_textlike
    4. 不对称增强: Y_out = Y + λ_bright * M * H_pos + λ_dark * M * H_neg
    5. 选择性暗描边 (仅 text-like 区域)
    """
    Y = np.clip(Y.astype(np.float32), 0.0, 1.0)

    # --- Sobel 梯度 ---
    Gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    Gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    grad_mag = np.sqrt(Gx * Gx + Gy * Gy)

    norm = percentile_fast(grad_mag, percentile=norm_percentile)
    norm = max(norm, 1e-6)
    grad_norm = np.clip(grad_mag / norm, 0.0, 1.0)

    # M_edge: 软边缘 mask
    M_edge = smoothstep(edge_low, edge_high, grad_norm).astype(np.float32)

    # --- 结构张量 coherence ---
    coherence = compute_coherence(Y, sigma_st=sigma_st)
    W_structure = smoothstep(coherence_low, coherence_high, coherence).astype(np.float32)

    # --- 文字/图标权重 ---
    W_textlike = compute_textlike_weight(
        Y, grad_norm, coherence,
        win=textlike_win,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
    )

    # --- 亮度门限 ---
    W_luma = smoothstep(luma_low, luma_high, Y).astype(np.float32)

    # --- 综合 mask ---
    M = M_edge * W_structure * W_luma
    M_text = M * W_textlike

    # --- High-pass detail ---
    Y_blur = cv2.GaussianBlur(
        Y, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT101
    )
    H = Y - Y_blur

    H_pos = np.maximum(H, 0.0)
    H_neg = np.minimum(H, 0.0)

    # --- 不对称增强 ---
    Y_out = Y + float(lambda_bright) * M * H_pos + float(lambda_dark) * M * H_neg

    # --- 选择性暗描边 (仅 text-like 区域，暗侧加深) ---
    if outline_dark > 0:
        Y_out = Y_out + float(outline_dark) * M_text * H_neg

    Y_out = np.clip(Y_out, 0.0, 1.0)

    return Y_out


# ============================================================
# 4. 统一入口
# ============================================================

def content_aware_edge_enhance_image(
    img,
    sigma=6.0,
    lambda_bright=2.0,
    lambda_dark=1.0,
    outline_dark=0.5,
    edge_low=0.02,
    edge_high=0.10,
    coherence_low=0.3,
    coherence_high=0.7,
    luma_low=0.02,
    luma_high=0.10,
    sigma_st=2.0,
    textlike_win=11,
    norm_percentile=99.5,
    standard="bt709",
):
    """
    统一入口。支持 Gray / BGR / BGRA。
    只调整 Y 通道，不修改色度。
    """
    if img is None:
        return None

    dtype = img.dtype
    max_value = get_dtype_max(img)

    kwargs = dict(
        sigma=sigma,
        lambda_bright=lambda_bright,
        lambda_dark=lambda_dark,
        outline_dark=outline_dark,
        edge_low=edge_low,
        edge_high=edge_high,
        coherence_low=coherence_low,
        coherence_high=coherence_high,
        luma_low=luma_low,
        luma_high=luma_high,
        sigma_st=sigma_st,
        textlike_win=textlike_win,
        norm_percentile=norm_percentile,
    )

    # Gray
    if img.ndim == 2:
        Y = img.astype(np.float32) / max_value
        Y = np.clip(Y, 0.0, 1.0)
        Y_out = content_aware_enhance_y(Y, **kwargs)
        return restore_dtype(Y_out, dtype)

    # BGR or BGRA
    if img.ndim == 3:
        c = img.shape[2]

        if c == 4:
            bgr = img[..., :3]
            alpha = img[..., 3:4]
        else:
            bgr = img
            alpha = None

        img_float = bgr.astype(np.float32) / max_value
        img_float = np.clip(img_float, 0.0, 1.0)

        Y, Cb, Cr = bgr_to_ycbcr_float(img_float, standard=standard)
        Y_out = content_aware_enhance_y(Y, **kwargs)
        out_float = ycbcr_to_bgr_float(Y_out, Cb, Cr, standard=standard, clip=True)

        out = restore_dtype(out_float, dtype)

        if alpha is not None:
            out = np.concatenate([out, alpha], axis=-1)

        return out

    raise ValueError(f"Unsupported image shape: {img.shape}")


# ============================================================
# 5. Lux 参数插值
# ============================================================

def _find_exact_key(value, keys, eps=1e-6):
    value = float(value)
    for key in keys:
        if abs(value - float(key)) <= eps:
            return key
    return None


def get_content_aware_params_by_lux(lux):
    """
    根据 lux 获取内容感知边缘增强参数。
    支持插值。
    """
    lux = float(lux)
    presets = CONTENT_AWARE_EDGE_PRESETS
    preset_lux = sorted(presets.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        return dict(presets[exact_key])

    if lux <= preset_lux[0]:
        return dict(presets[preset_lux[0]])

    if lux >= preset_lux[-1]:
        return dict(presets[preset_lux[-1]])

    # 线性插值
    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]
        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)
            p0 = presets[l0]
            p1 = presets[l1]
            out = {}
            for key in p0.keys():
                if isinstance(p0[key], (int, float)):
                    out[key] = p0[key] * (1.0 - t) + p1[key] * t
                else:
                    out[key] = p0[key]
            return out

    return dict(presets[preset_lux[-1]])
