# sobel_edge_enhance.py

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

from instrument.config import EDGE_PRESETS


# ============================================================
# 1. Sobel 边缘增强参数
# ============================================================


def _find_exact_key(value, keys, eps=1e-6):
    value = float(value)

    for key in keys:
        if abs(value - float(key)) <= eps:
            return key

    return None


def _sanitize_edge_params(params):
    """
    填默认值并做安全裁剪。
    """
    out = {
        "amount": 0.35,
        "sigma": 0.90,
        "outline_strength": 0.0,
        "edge_low": 0.10,
        "edge_high": 0.34,
        "edge_blur": 0.80,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
        "high_luma_low": 0.82,
        "high_luma_high": 0.96,
        "high_luma_suppress": 0.0,
    }

    out.update(dict(params))

    out["amount"] = float(np.clip(out["amount"], 0.0, 5.0))
    out["sigma"] = float(max(out["sigma"], 0.0))

    out["edge_low"] = float(np.clip(out["edge_low"], 0.0, 1.0))
    out["edge_high"] = float(np.clip(out["edge_high"], 0.0, 1.0))

    if out["edge_high"] <= out["edge_low"]:
        out["edge_high"] = min(out["edge_low"] + 1e-3, 1.0)

    out["edge_blur"] = float(max(out["edge_blur"], 0.0))

    out["luma_low"] = float(np.clip(out["luma_low"], 0.0, 1.0))
    out["luma_high"] = float(np.clip(out["luma_high"], 0.0, 1.0))

    if out["luma_high"] <= out["luma_low"]:
        out["luma_high"] = min(out["luma_low"] + 1e-3, 1.0)

    out["norm_percentile"] = float(np.clip(out["norm_percentile"], 90.0, 99.99))

    out["high_luma_low"] = float(np.clip(out["high_luma_low"], 0.0, 1.0))
    out["high_luma_high"] = float(np.clip(out["high_luma_high"], 0.0, 1.0))

    if out["high_luma_high"] <= out["high_luma_low"]:
        out["high_luma_high"] = min(out["high_luma_low"] + 1e-3, 1.0)

    out["high_luma_suppress"] = float(np.clip(out["high_luma_suppress"], 0.0, 1.0))

    out["outline_strength"] = float(np.clip(out.get("outline_strength", 0.0), 0.0, 1.0))

    return out


def get_edge_params_by_lux(lux):
    """
    根据 lux 获取 Sobel 边缘增强参数。
    如果 lux 不在 preset 中，则线性插值。
    """
    lux = float(lux)
    preset_lux = sorted(EDGE_PRESETS.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        return dict(EDGE_PRESETS[exact_key])

    if lux <= preset_lux[0]:
        return dict(EDGE_PRESETS[preset_lux[0]])

    if lux >= preset_lux[-1]:
        return dict(EDGE_PRESETS[preset_lux[-1]])

    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)

            p0 = EDGE_PRESETS[l0]
            p1 = EDGE_PRESETS[l1]

            out = {}
            for key in p0.keys():
                out[key] = p0[key] * (1.0 - t) + p1[key] * t

            return out

    raise RuntimeError("Unexpected edge parameter interpolation error.")


def extract_y_from_image_float(img, standard="bt709"):
    """
    从 Gray/BGR/BGRA 图像中提取 Y，返回 float [0,1]。
    """
    return compute_luma_float(img, standard=standard).astype(np.float32)


def compute_luma_stats_from_y(Y, max_samples=1000000):
    """
    基于 Y 通道统计亮度分布。
    """
    Y = np.clip(Y.astype(np.float32), 0.0, 1.0)

    flat = Y.reshape(-1)

    if flat.size > max_samples:
        step = max(1, flat.size // max_samples)
        flat = flat[::step]

    p5, p25, p50, p75, p95 = np.percentile(
        flat,
        [5, 25, 50, 75, 95],
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

    return stats


def blend_param(a, b, w):
    return float(a) * (1.0 - float(w)) + float(b) * float(w)


def adapt_edge_params_by_content(Y, params):
    """
    根据画面亮度分布修正 Sobel 边缘增强参数。
    """
    stats = compute_luma_stats_from_y(Y)

    out = _sanitize_edge_params(params)

    amount = float(out.get("amount", 0.35))

    # 当前 preset 中 amount 大致 1.50 ~ 3.50。
    lux_strength = np.clip((amount - 1.50) / 2.00, 0.0, 1.0)

    high_key_score = (
        smoothstep(0.55, 0.80, stats["r_high"])
        * smoothstep(0.58, 0.72, stats["p50"])
    )
    high_key_score = float(np.clip(high_key_score, 0.0, 1.0))

    dark_score = (
        smoothstep(0.45, 0.70, stats["r_low"])
        * (1.0 - smoothstep(0.32, 0.50, stats["p50"]))
    )
    dark_score = float(np.clip(dark_score, 0.0, 1.0))

    if high_key_score > 0.0:
        w = high_key_score

        target_amount = 1.50 + 1.00 * lux_strength

        out["amount"] = blend_param(out["amount"], target_amount, w)
        out["sigma"] = blend_param(out["sigma"], 0.80, w)
        out["outline_strength"] = blend_param(
            out["outline_strength"], out["outline_strength"] * 0.85, w
        )

        out["edge_low"] = blend_param(out["edge_low"], 0.045, w)
        out["edge_high"] = blend_param(out["edge_high"], 0.22, w)
        out["edge_blur"] = blend_param(out["edge_blur"], 0.45, w)

        out["luma_low"] = blend_param(out["luma_low"], 0.00, w)
        out["luma_high"] = blend_param(out["luma_high"], 0.04, w)

        out["high_luma_low"] = 0.82
        out["high_luma_high"] = 0.96
        out["high_luma_suppress"] = max(
            float(out.get("high_luma_suppress", 0.0)),
            0.25 + 0.45 * w,
        )

    if dark_score > 0.0:
        w = dark_score

        target_amount = 1.00 + 0.80 * lux_strength

        out["amount"] = blend_param(out["amount"], target_amount, w)
        out["outline_strength"] = blend_param(
            out["outline_strength"], out["outline_strength"] * 0.75, w
        )
        out["edge_low"] = blend_param(out["edge_low"], 0.12, w)
        out["edge_high"] = blend_param(out["edge_high"], 0.36, w)

        out["luma_low"] = blend_param(out["luma_low"], 0.05, w)
        out["luma_high"] = blend_param(out["luma_high"], 0.14, w)
        out["edge_blur"] = blend_param(out["edge_blur"], 0.80, w)

    out = _sanitize_edge_params(out)

    return out, stats


def prepare_edge_params_for_image(
    img,
    params,
    standard="bt709",
    content_adaptive=True,
):
    """
    获取当前图像实际使用的 edge 参数。

    返回：
        edge_params_eff, luma_stats
    """
    params = _sanitize_edge_params(params)

    if not content_adaptive:
        return params, None

    Y = extract_y_from_image_float(img, standard=standard)
    return adapt_edge_params_by_content(Y, params)


def enhance_y_by_sobel(
    Y,
    amount=0.35,
    sigma=0.9,
    outline_strength=0.0,
    edge_low=0.10,
    edge_high=0.34,
    edge_blur=0.8,
    luma_low=0.02,
    luma_high=0.10,
    norm_percentile=99.5,
    sobel_ksize=3,
    high_luma_low=0.82,
    high_luma_high=0.96,
    high_luma_suppress=0.0,
):
    """
    只在 Y 通道上进行 Sobel 边缘检测 + 边缘锐化。
    """
    Y = np.clip(Y.astype(np.float32), 0.0, 1.0)

    gx = cv2.Sobel(
        Y,
        cv2.CV_32F,
        1,
        0,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REFLECT101,
    )

    gy = cv2.Sobel(
        Y,
        cv2.CV_32F,
        0,
        1,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REFLECT101,
    )

    grad_mag = np.sqrt(gx * gx + gy * gy)

    norm = percentile_fast(grad_mag, percentile=norm_percentile)
    norm = max(norm, 1e-6)

    grad_norm = np.clip(grad_mag / norm, 0.0, 1.0)

    edge_weight = smoothstep(edge_low, edge_high, grad_norm).astype(np.float32)

    if edge_blur is not None and edge_blur > 0:
        edge_weight = cv2.GaussianBlur(
            edge_weight,
            ksize=(0, 0),
            sigmaX=float(edge_blur),
            sigmaY=float(edge_blur),
            borderType=cv2.BORDER_REFLECT101,
        )

    luma_gate = smoothstep(luma_low, luma_high, Y)
    edge_weight = edge_weight * luma_gate

    if sigma is not None and sigma > 0:
        Y_blur = cv2.GaussianBlur(
            Y,
            ksize=(0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT101,
        )
    else:
        Y_blur = Y

    detail = Y - Y_blur

    if high_luma_suppress is not None and high_luma_suppress > 0:
        high_w = smoothstep(high_luma_low, high_luma_high, Y)
        pos_scale = 1.0 - float(high_luma_suppress) * high_w
        pos_scale = np.clip(pos_scale, 0.0, 1.0)

        detail = np.where(detail > 0.0, detail * pos_scale, detail)

    # 传统 unsharp mask 增强
    Y_out = Y + float(amount) * edge_weight * detail

    # 高反差描边：用固定幅度 + 梯度方向画黑白勾边
    if outline_strength is not None and outline_strength > 0:
        edge_sign = np.sign(detail)
        Y_out = Y_out + float(outline_strength) * edge_weight * edge_sign

    Y_out = np.clip(Y_out, 0.0, 1.0)

    return Y_out


def sobel_edge_enhance_bgr_y_only(
    img,
    amount=0.35,
    sigma=0.9,
    outline_strength=0.0,
    edge_low=0.10,
    edge_high=0.34,
    edge_blur=0.8,
    luma_low=0.02,
    luma_high=0.10,
    norm_percentile=99.5,
    standard="bt709",
    high_luma_low=0.82,
    high_luma_high=0.96,
    high_luma_suppress=0.0,
):
    """
    输入 BGR 图像，只调整 Y 通道，保持 Cb/Cr 不变。
    """
    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    Y, Cb, Cr = bgr_to_ycbcr_float(img_float, standard=standard)

    Y_out = enhance_y_by_sobel(
        Y,
        amount=amount,
        sigma=sigma,
        outline_strength=outline_strength,
        edge_low=edge_low,
        edge_high=edge_high,
        edge_blur=edge_blur,
        luma_low=luma_low,
        luma_high=luma_high,
        norm_percentile=norm_percentile,
        high_luma_low=high_luma_low,
        high_luma_high=high_luma_high,
        high_luma_suppress=high_luma_suppress,
    )

    out_float = ycbcr_to_bgr_float(Y_out, Cb, Cr, standard=standard, clip=True)

    return restore_dtype(out_float, dtype)


def sobel_edge_enhance_gray(
    img,
    amount=0.35,
    sigma=0.9,
    outline_strength=0.0,
    edge_low=0.10,
    edge_high=0.34,
    edge_blur=0.8,
    luma_low=0.02,
    luma_high=0.10,
    norm_percentile=99.5,
    high_luma_low=0.82,
    high_luma_high=0.96,
    high_luma_suppress=0.0,
):
    """
    灰度图直接在单通道上做 Sobel 边缘锐化。
    """
    dtype = img.dtype
    max_value = get_dtype_max(img)

    Y = img.astype(np.float32) / max_value
    Y = np.clip(Y, 0.0, 1.0)

    Y_out = enhance_y_by_sobel(
        Y,
        amount=amount,
        sigma=sigma,
        outline_strength=outline_strength,
        edge_low=edge_low,
        edge_high=edge_high,
        edge_blur=edge_blur,
        luma_low=luma_low,
        luma_high=luma_high,
        norm_percentile=norm_percentile,
        high_luma_low=high_luma_low,
        high_luma_high=high_luma_high,
        high_luma_suppress=high_luma_suppress,
    )

    return restore_dtype(Y_out, dtype)


def sobel_edge_enhance_image(
    img,
    amount=0.35,
    sigma=0.9,
    outline_strength=0.0,
    edge_low=0.10,
    edge_high=0.34,
    edge_blur=0.8,
    luma_low=0.02,
    luma_high=0.10,
    norm_percentile=99.5,
    standard="bt709",
    high_luma_low=0.82,
    high_luma_high=0.96,
    high_luma_suppress=0.0,
    content_adaptive=True,
    return_info=False,
):
    """
    通用接口。

    支持：
        - Gray
        - BGR
        - BGRA

    只调整亮度 Y，不主动修改色度。
    """
    if img is None:
        return None

    params = {
        "amount": amount,
        "sigma": sigma,
        "outline_strength": outline_strength,
        "edge_low": edge_low,
        "edge_high": edge_high,
        "edge_blur": edge_blur,
        "luma_low": luma_low,
        "luma_high": luma_high,
        "norm_percentile": norm_percentile,
        "high_luma_low": high_luma_low,
        "high_luma_high": high_luma_high,
        "high_luma_suppress": high_luma_suppress,
    }

    params, luma_stats = prepare_edge_params_for_image(
        img,
        params,
        standard=standard,
        content_adaptive=content_adaptive,
    )

    if img.ndim == 2:
        out = sobel_edge_enhance_gray(
            img,
            amount=params["amount"],
            sigma=params["sigma"],
            outline_strength=params["outline_strength"],
            edge_low=params["edge_low"],
            edge_high=params["edge_high"],
            edge_blur=params["edge_blur"],
            luma_low=params["luma_low"],
            luma_high=params["luma_high"],
            norm_percentile=params["norm_percentile"],
            high_luma_low=params["high_luma_low"],
            high_luma_high=params["high_luma_high"],
            high_luma_suppress=params["high_luma_suppress"],
        )

        if return_info:
            return out, {
                "edge_params": params,
                "luma_stats": luma_stats,
            }

        return out

    if img.ndim == 3:
        c = img.shape[2]

        if c == 3:
            out = sobel_edge_enhance_bgr_y_only(
                img,
                amount=params["amount"],
                sigma=params["sigma"],
                outline_strength=params["outline_strength"],
                edge_low=params["edge_low"],
                edge_high=params["edge_high"],
                edge_blur=params["edge_blur"],
                luma_low=params["luma_low"],
                luma_high=params["luma_high"],
                norm_percentile=params["norm_percentile"],
                standard=standard,
                high_luma_low=params["high_luma_low"],
                high_luma_high=params["high_luma_high"],
                high_luma_suppress=params["high_luma_suppress"],
            )

            if return_info:
                return out, {
                    "edge_params": params,
                    "luma_stats": luma_stats,
                }

            return out

        if c == 4:
            bgr = img[..., :3]
            alpha = img[..., 3:4]

            bgr_out = sobel_edge_enhance_bgr_y_only(
                bgr,
                amount=params["amount"],
                sigma=params["sigma"],
                outline_strength=params["outline_strength"],
                edge_low=params["edge_low"],
                edge_high=params["edge_high"],
                edge_blur=params["edge_blur"],
                luma_low=params["luma_low"],
                luma_high=params["luma_high"],
                norm_percentile=params["norm_percentile"],
                standard=standard,
                high_luma_low=params["high_luma_low"],
                high_luma_high=params["high_luma_high"],
                high_luma_suppress=params["high_luma_suppress"],
            )

            out = np.concatenate([bgr_out, alpha], axis=-1)

            if return_info:
                return out, {
                    "edge_params": params,
                    "luma_stats": luma_stats,
                }

            return out

    raise ValueError(f"Unsupported image shape for Sobel edge enhance: {img.shape}")


def compute_sobel_debug_from_y(
    Y,
    amount=0.35,
    sigma=0.9,
    edge_low=0.10,
    edge_high=0.34,
    edge_blur=0.8,
    luma_low=0.02,
    luma_high=0.10,
    norm_percentile=99.5,
    sobel_ksize=3,
    high_luma_low=0.82,
    high_luma_high=0.96,
    high_luma_suppress=0.0,
):
    """
    基于 Y 通道计算 Sobel 边缘增强相关调试信息。
    """
    Y = np.clip(Y.astype(np.float32), 0.0, 1.0)

    gx = cv2.Sobel(
        Y,
        cv2.CV_32F,
        1,
        0,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REFLECT101,
    )

    gy = cv2.Sobel(
        Y,
        cv2.CV_32F,
        0,
        1,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REFLECT101,
    )

    grad_mag = np.sqrt(gx * gx + gy * gy)

    norm = percentile_fast(grad_mag, percentile=norm_percentile)
    norm = max(norm, 1e-6)

    grad_norm = np.clip(grad_mag / norm, 0.0, 1.0)

    edge_weight = smoothstep(edge_low, edge_high, grad_norm).astype(np.float32)

    if edge_blur is not None and edge_blur > 0:
        edge_weight = cv2.GaussianBlur(
            edge_weight,
            ksize=(0, 0),
            sigmaX=float(edge_blur),
            sigmaY=float(edge_blur),
            borderType=cv2.BORDER_REFLECT101,
        )

    luma_gate = smoothstep(luma_low, luma_high, Y)
    edge_weight = edge_weight * luma_gate

    if sigma is not None and sigma > 0:
        Y_blur = cv2.GaussianBlur(
            Y,
            ksize=(0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT101,
        )
    else:
        Y_blur = Y

    detail_raw = Y - Y_blur
    detail = detail_raw.copy()

    high_w = np.zeros_like(Y, dtype=np.float32)

    if high_luma_suppress is not None and high_luma_suppress > 0:
        high_w = smoothstep(high_luma_low, high_luma_high, Y).astype(np.float32)
        pos_scale = 1.0 - float(high_luma_suppress) * high_w
        pos_scale = np.clip(pos_scale, 0.0, 1.0)

        detail = np.where(detail > 0.0, detail * pos_scale, detail)

    detail_abs = np.abs(detail)

    enhanced_detail = float(amount) * edge_weight * detail
    Y_out = np.clip(Y + enhanced_detail, 0.0, 1.0)

    tenengrad = float(np.mean(gx * gx + gy * gy))
    grad_mean = float(np.mean(grad_mag))
    grad_p95 = float(percentile_fast(grad_mag, 95))
    grad_p99 = float(percentile_fast(grad_mag, 99))

    edge_ratio_01 = float(np.mean(edge_weight > 0.1))
    edge_ratio_03 = float(np.mean(edge_weight > 0.3))
    edge_weight_mean = float(np.mean(edge_weight))
    edge_weight_p95 = float(percentile_fast(edge_weight, 95))

    detail_abs_255 = float(np.mean(detail_abs) * 255.0)
    detail_abs_p95_255 = float(percentile_fast(detail_abs, 95) * 255.0)

    enhanced_abs = np.abs(enhanced_detail)
    enhanced_abs_mean_255 = float(np.mean(enhanced_abs) * 255.0)
    enhanced_abs_p95_255 = float(percentile_fast(enhanced_abs, 95) * 255.0)

    luma_stats = compute_luma_stats_from_y(Y)

    metrics = {
        "amount": float(amount),
        "sigma": float(sigma),
        "edge_low": float(edge_low),
        "edge_high": float(edge_high),
        "edge_blur": float(edge_blur),
        "luma_low": float(luma_low),
        "luma_high": float(luma_high),
        "norm_percentile": float(norm_percentile),
        "high_luma_low": float(high_luma_low),
        "high_luma_high": float(high_luma_high),
        "high_luma_suppress": float(high_luma_suppress),

        "tenengrad": tenengrad,
        "grad_mean": grad_mean,
        "grad_p95": grad_p95,
        "grad_p99": grad_p99,

        "edge_ratio_01": edge_ratio_01,
        "edge_ratio_03": edge_ratio_03,
        "edge_weight_mean": edge_weight_mean,
        "edge_weight_p95": edge_weight_p95,

        "detail_abs_mean_255": detail_abs_255,
        "detail_abs_p95_255": detail_abs_p95_255,

        "enhanced_abs_mean_255": enhanced_abs_mean_255,
        "enhanced_abs_p95_255": enhanced_abs_p95_255,

        "luma_p5": luma_stats["p5"],
        "luma_p50": luma_stats["p50"],
        "luma_p95": luma_stats["p95"],
        "luma_r_high": luma_stats["r_high"],
        "luma_r_low": luma_stats["r_low"],
    }

    debug = {
        "grad_mag": grad_mag,
        "grad_norm": grad_norm,
        "edge_weight": edge_weight,
        "luma_gate": luma_gate,
        "high_w": high_w,
        "detail_raw": detail_raw,
        "detail": detail,
        "detail_abs": detail_abs,
        "enhanced_detail": enhanced_detail,
        "Y_out": Y_out,
        "metrics": metrics,
    }

    return debug


def compute_sobel_debug_image(
    img,
    amount=0.35,
    sigma=0.9,
    edge_low=0.10,
    edge_high=0.34,
    edge_blur=0.8,
    luma_low=0.02,
    luma_high=0.10,
    norm_percentile=99.5,
    standard="bt709",
    high_luma_low=0.82,
    high_luma_high=0.96,
    high_luma_suppress=0.0,
    content_adaptive=True,
):
    """
    对输入图像计算 Sobel 边缘增强调试图和量化指标。
    不修改原图，只用于对比图可视化和指标分析。
    """
    params = {
        "amount": amount,
        "sigma": sigma,
        "edge_low": edge_low,
        "edge_high": edge_high,
        "edge_blur": edge_blur,
        "luma_low": luma_low,
        "luma_high": luma_high,
        "norm_percentile": norm_percentile,
        "high_luma_low": high_luma_low,
        "high_luma_high": high_luma_high,
        "high_luma_suppress": high_luma_suppress,
    }

    params, stats = prepare_edge_params_for_image(
        img,
        params,
        standard=standard,
        content_adaptive=content_adaptive,
    )

    Y = extract_y_from_image_float(img, standard=standard)

    debug = compute_sobel_debug_from_y(
        Y,
        amount=params["amount"],
        sigma=params["sigma"],
        edge_low=params["edge_low"],
        edge_high=params["edge_high"],
        edge_blur=params["edge_blur"],
        luma_low=params["luma_low"],
        luma_high=params["luma_high"],
        norm_percentile=params["norm_percentile"],
        high_luma_low=params["high_luma_low"],
        high_luma_high=params["high_luma_high"],
        high_luma_suppress=params["high_luma_suppress"],
    )

    debug["edge_params"] = params
    debug["content_stats"] = stats

    return debug