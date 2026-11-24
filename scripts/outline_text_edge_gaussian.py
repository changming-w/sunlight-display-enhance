#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text Edge Gaussian-Smoothed Outline.

功能：
    1. 纯传统视觉方法，不依赖 OCR / 深度模型；
    2. 先生成文字 seed mask；
    3. 在文字 seed 附近做 Canny + Scharr 边缘检测；
    4. 不做最终膨胀，不做 distance transform；
    5. 直接对 edge mask 做 GaussianBlur，得到平滑 alpha；
    6. 用 alpha blend 给文字边缘上色，减少锯齿；
    7. 支持单图 / 文件夹批量；
    8. 支持 debug mask 输出。

依赖：
    pip install opencv-python numpy

推荐运行：
    python scripts/outline_text_edge_gaussian.py \
        --input ./test/simulate_sunlight/enhanced_80000lux \
        --output ./test/outline_demo \
        --recursive \
        --stroke-radius 0 \
        --stroke-alpha 0.9 \
        --edge-dilate 0 \
        --edge-close 1 \
        --aa-sigma 0.65 \
        --save-debug
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# =============================================================================
# IO
# =============================================================================

def imread_unicode(path: str) -> Optional[np.ndarray]:
    """支持中文路径读图。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """支持中文路径写图。"""
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        ext = path_obj.suffix
        if ext == "":
            ext = ".png"

        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False

        buf.tofile(str(path_obj))
        return True
    except Exception:
        return False


def collect_images(input_path: str, recursive: bool = False) -> List[Path]:
    p = Path(input_path)

    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []

    if not p.exists() or not p.is_dir():
        return []

    if recursive:
        paths = [
            x for x in p.rglob("*")
            if x.is_file() and x.suffix.lower() in IMAGE_EXTS
        ]
    else:
        paths = [
            x for x in p.glob("*")
            if x.is_file() and x.suffix.lower() in IMAGE_EXTS
        ]

    return sorted(paths)


def get_relative_output_path(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    suffix: str = "_edge_gaussian",
) -> Path:
    if input_root.is_file():
        rel = Path(image_path.name)
    else:
        try:
            rel = image_path.relative_to(input_root)
        except Exception:
            rel = Path(image_path.name)

    out_name = rel.stem + suffix + ".png"
    return output_root / rel.parent / out_name


# =============================================================================
# dtype / channel utilities
# =============================================================================

def dtype_max(img: np.ndarray) -> float:
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    if np.issubdtype(img.dtype, np.floating):
        return 1.0
    raise TypeError(f"Unsupported dtype: {img.dtype}")


def to_float01(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        out = img.astype(np.float32) / dtype_max(img)
    else:
        out = img.astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def restore_dtype(img_float: np.ndarray, dtype: np.dtype) -> np.ndarray:
    img_float = np.clip(img_float, 0.0, 1.0)

    if np.issubdtype(dtype, np.integer):
        maxv = float(np.iinfo(dtype).max)
        return np.round(img_float * maxv).clip(0, maxv).astype(dtype)

    return img_float.astype(dtype)


def to_uint8_for_detection(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.copy()

    f = to_float01(img)
    return np.round(f * 255.0).clip(0, 255).astype(np.uint8)


def split_bgr_alpha(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    返回 BGR 和 alpha。
    灰度图会转 BGR。
    """
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, None

    if img.ndim == 3:
        if img.shape[2] == 4:
            return img[:, :, :3], img[:, :, 3]
        if img.shape[2] == 3:
            return img, None

    raise ValueError(f"Unsupported image shape: {img.shape}")


def merge_bgr_alpha(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    if alpha is None:
        return bgr
    return np.dstack([bgr, alpha])


# =============================================================================
# Luma / bbox
# =============================================================================

def bgr_float_to_luma01(bgr_float: np.ndarray) -> np.ndarray:
    """
    BGR float [0,1] -> BT.709 luma [0,1]
    """
    B = bgr_float[..., 0]
    G = bgr_float[..., 1]
    R = bgr_float[..., 2]

    Y = 0.0722 * B + 0.7152 * G + 0.2126 * R
    return np.clip(Y, 0.0, 1.0).astype(np.float32)


def bbox_from_mask(
    mask: np.ndarray,
    pad: int,
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)

    if xs.size == 0 or ys.size == 0:
        return None

    h, w = image_shape[:2]

    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)

    return x0, y0, x1, y1


# =============================================================================
# Color utilities
# =============================================================================

def median_color_bgr_float(
    img_float: np.ndarray,
    mask: np.ndarray,
    default: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算 mask 区域 BGR float [0,1] 中位颜色。
    """
    if default is None:
        default = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    m = mask.astype(bool)
    if m.sum() == 0:
        return default.astype(np.float32)

    pixels = img_float[m]
    if pixels.size == 0:
        return default.astype(np.float32)

    color = np.median(pixels.reshape(-1, 3), axis=0)
    return np.clip(color, 0.0, 1.0).astype(np.float32)


def bgr_float_to_hsv_cv(color_bgr: np.ndarray) -> np.ndarray:
    """
    BGR float [0,1] -> OpenCV HSV:
        H: 0~179
        S: 0~255
        V: 0~255
    """
    color_bgr = np.asarray(color_bgr, dtype=np.float32).reshape(3)
    color_u8 = np.round(np.clip(color_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    patch = color_u8.reshape(1, 1, 3)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
    return hsv.astype(np.float32)


def hsv_cv_to_bgr_float(h: float, s: float, v: float) -> np.ndarray:
    """
    OpenCV HSV -> BGR float [0,1]
    """
    hh = int(round(h)) % 180
    ss = int(round(np.clip(s, 0, 255)))
    vv = int(round(np.clip(v, 0, 255)))

    hsv = np.array([[[hh, ss, vv]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255.0
    return np.clip(bgr, 0.0, 1.0).astype(np.float32)


def relative_luminance_srgb_bgr(color_bgr: np.ndarray) -> float:
    """
    WCAG 相对亮度。
    输入 BGR float [0,1]。
    """
    color_bgr = np.asarray(color_bgr, dtype=np.float32).reshape(3)
    rgb = np.clip(color_bgr[::-1], 0.0, 1.0)

    def linearize(c: np.ndarray) -> np.ndarray:
        return np.where(
            c <= 0.03928,
            c / 12.92,
            ((c + 0.055) / 1.055) ** 2.4,
        )

    r, g, b = linearize(rgb)
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return float(L)


def contrast_ratio_bgr(c1: np.ndarray, c2: np.ndarray) -> float:
    L1 = relative_luminance_srgb_bgr(c1)
    L2 = relative_luminance_srgb_bgr(c2)

    Lh = max(L1, L2)
    Ll = min(L1, L2)

    return float((Lh + 0.05) / (Ll + 0.05))


def hue_distance_score(candidate_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    """
    衡量 candidate 是否接近 ref 的互补色。
    返回 [0,1]，越大越接近互补。
    OpenCV Hue 0~179，所以互补色相差 90。
    """
    hsv_c = bgr_float_to_hsv_cv(candidate_bgr)
    hsv_r = bgr_float_to_hsv_cv(ref_bgr)

    if hsv_r[1] < 25:
        return 0.0

    h_c = float(hsv_c[0])
    h_r = float(hsv_r[0])
    h_comp = (h_r + 90.0) % 180.0

    d = abs(h_c - h_comp)
    d = min(d, 180.0 - d)

    return float(np.clip(1.0 - d / 90.0, 0.0, 1.0))


def build_complement_candidates(
    fg_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    min_sat: int = 150,
) -> List[np.ndarray]:
    """
    构造互补色候选 + 安全高对比候选。
    """
    candidates: List[np.ndarray] = []

    for ref in [fg_bgr, bg_bgr]:
        hsv = bgr_float_to_hsv_cv(ref)
        h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

        if s >= 25:
            h_comp = (h + 90.0) % 180.0
            s_comp = max(s, float(min_sat))

            candidates.append(hsv_cv_to_bgr_float(h_comp, s_comp, 245))
            candidates.append(hsv_cv_to_bgr_float(h_comp, s_comp, 190))
            candidates.append(hsv_cv_to_bgr_float(h_comp, s_comp, 80))
            candidates.append(hsv_cv_to_bgr_float(h_comp, 230, 230))

    # BGR float 安全色
    safe_colors = [
        [0.0, 0.0, 0.0],       # black
        [1.0, 1.0, 1.0],       # white
        [0.0, 1.0, 1.0],       # yellow
        [1.0, 0.0, 0.0],       # blue
        [1.0, 1.0, 0.0],       # cyan
        [1.0, 0.0, 1.0],       # magenta
        [0.0, 0.5, 1.0],       # orange-ish
    ]

    for c in safe_colors:
        candidates.append(np.asarray(c, dtype=np.float32))

    unique: List[np.ndarray] = []

    for c in candidates:
        c = np.clip(c, 0.0, 1.0).astype(np.float32)

        duplicated = False
        for u in unique:
            if np.linalg.norm(c - u) < 1e-3:
                duplicated = True
                break

        if not duplicated:
            unique.append(c)

    return unique


def choose_stroke_color(
    fg_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    prefer_ref_bgr: np.ndarray,
    min_bg_contrast: float = 3.0,
    min_fg_contrast: float = 1.1,
    min_sat: int = 150,
) -> np.ndarray:
    """
    选择边缘描色：
        1. 背景对比度最重要；
        2. 互补色作为偏好；
        3. 黑白安全色兜底。
    """
    candidates = build_complement_candidates(
        fg_bgr=fg_bgr,
        bg_bgr=bg_bgr,
        min_sat=min_sat,
    )

    best_color = candidates[0]
    best_score = -1e9

    for c in candidates:
        cr_bg = contrast_ratio_bgr(c, bg_bgr)
        cr_fg = contrast_ratio_bgr(c, fg_bgr)
        hue_score = hue_distance_score(c, prefer_ref_bgr)

        hsv_c = bgr_float_to_hsv_cv(c)
        sat_score = float(hsv_c[1]) / 255.0

        score = (
            3.0 * min(cr_bg, 8.0)
            + 0.6 * min(cr_fg, 5.0)
            + 1.0 * hue_score
            + 0.2 * sat_score
        )

        if cr_bg < min_bg_contrast:
            score -= 10.0 * (min_bg_contrast - cr_bg)

        if cr_fg < min_fg_contrast:
            score -= 2.0 * (min_fg_contrast - cr_fg)

        if score > best_score:
            best_score = score
            best_color = c

    if contrast_ratio_bgr(best_color, bg_bgr) < min_bg_contrast:
        black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        cb = contrast_ratio_bgr(black, bg_bgr)
        cw = contrast_ratio_bgr(white, bg_bgr)

        best_color = black if cb >= cw else white

    return np.clip(best_color, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Mask utilities
# =============================================================================

def safe_odd(k: int) -> int:
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k


def make_outer_context_ring(
    mask: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """
    用于估计文字周围背景颜色，不是最终绘制。
    """
    m = (mask > 0).astype(np.uint8)

    inner_radius = max(1, int(inner_radius))
    outer_radius = max(inner_radius + 1, int(outer_radius))

    k_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * inner_radius + 1, 2 * inner_radius + 1),
    )

    k_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * outer_radius + 1, 2 * outer_radius + 1),
    )

    dil_inner = cv2.dilate(m, k_inner)
    dil_outer = cv2.dilate(m, k_outer)

    ring = np.logical_and(dil_outer > 0, dil_inner == 0)

    return ring.astype(np.uint8) * 255


def group_mask_instances(
    mask: np.ndarray,
    group_kernel_w: int,
    group_kernel_h: int,
    min_group_area: int,
) -> List[np.ndarray]:
    """
    将文字 seed 分组。
    注意：
        分组只用于确定 ROI 和颜色统计。
        最终绘制仍然只画 edge alpha。
    """
    if mask is None or mask.max() == 0:
        return []

    k = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(1, group_kernel_w), max(1, group_kernel_h)),
    )

    grouped = cv2.dilate((mask > 0).astype(np.uint8), k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        grouped,
        connectivity=8,
    )

    instances: List[np.ndarray] = []
    src_bool = mask > 0

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_group_area:
            continue

        group_region = labels == i
        inst = np.logical_and(src_bool, group_region)

        if int(inst.sum()) < min_group_area:
            continue

        instances.append(inst.astype(np.uint8) * 255)

    return instances


# =============================================================================
# Text seed detection
# =============================================================================

def detect_text_seed_mask_traditional(
    bgr_u8: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    传统方法生成文字 seed mask。
    这个 seed 只用于限制后续边缘检测范围，不直接绘制。
    """
    h, w = bgr_u8.shape[:2]
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)

    if args.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=args.clahe_clip,
            tileGridSize=(8, 8),
        )
        gray_work = clahe.apply(gray)
    else:
        gray_work = gray

    # 1. 局部对比
    sigma = max(float(args.text_local_sigma), 1.0)
    bg = cv2.GaussianBlur(gray_work, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = cv2.absdiff(gray_work, bg)

    _, diff_mask = cv2.threshold(
        diff,
        int(args.text_diff_thresh),
        255,
        cv2.THRESH_BINARY,
    )

    # 2. 梯度
    gx = cv2.Sobel(gray_work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_work, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = np.clip(grad, 0, 255).astype(np.uint8)

    _, grad_mask = cv2.threshold(
        grad,
        int(args.text_grad_thresh),
        255,
        cv2.THRESH_BINARY,
    )

    # 3. top-hat / black-hat
    k_text = safe_odd(args.text_morph_kernel)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_text, k_text))

    tophat = cv2.morphologyEx(gray_work, cv2.MORPH_TOPHAT, morph_kernel)
    blackhat = cv2.morphologyEx(gray_work, cv2.MORPH_BLACKHAT, morph_kernel)

    response = np.maximum(tophat, blackhat)

    resp_thr = max(
        float(args.text_response_thresh),
        float(np.mean(response) + args.text_response_std_scale * np.std(response)),
    )

    resp_mask = (response >= resp_thr).astype(np.uint8) * 255

    candidate = cv2.bitwise_or(
        cv2.bitwise_and(diff_mask, grad_mask),
        cv2.bitwise_and(resp_mask, grad_mask),
    )

    # 4. 可选 MSER
    if args.use_mser:
        mser_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_work)

            for pts in regions:
                if pts is None or len(pts) == 0:
                    continue

                x, y, bw, bh = cv2.boundingRect(pts)
                area = float(len(pts))
                box_area = float(max(bw * bh, 1))
                fill = area / box_area
                aspect = bw / max(float(bh), 1.0)

                if bh < args.text_min_height or bh > args.text_max_height:
                    continue
                if bw < args.text_min_width:
                    continue
                if area < args.text_min_area or area > args.text_max_area:
                    continue
                if fill < 0.02 or fill > 0.90:
                    continue
                if aspect > args.text_max_aspect:
                    continue

                cv2.fillPoly(mser_mask, [pts.reshape(-1, 1, 2)], 255)

            candidate = cv2.bitwise_or(candidate, mser_mask)

        except Exception:
            pass

    # 5. 清理
    if args.text_cleanup_open > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (args.text_cleanup_open, args.text_cleanup_open),
        )
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open)

    if args.text_cleanup_close > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (args.text_cleanup_close, args.text_cleanup_close),
        )
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close)

    # 6. 连通域过滤
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        connectivity=8,
    )

    seed_mask = np.zeros((h, w), dtype=np.uint8)
    image_area = h * w

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]

        if area < args.text_min_area:
            continue
        if area > min(args.text_max_area, image_area * 0.03):
            continue
        if bh < args.text_min_height or bh > args.text_max_height:
            continue
        if bw < args.text_min_width:
            continue

        aspect = bw / max(float(bh), 1.0)
        fill = area / max(float(bw * bh), 1.0)

        if fill < args.text_min_fill or fill > args.text_max_fill:
            continue
        if aspect > args.text_max_aspect:
            continue

        comp = labels == i
        seed_mask[comp] = 255

    # 默认不建议打开。这里只影响 seed，不是最终绘制。
    if args.text_final_dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.text_final_dilate + 1, 2 * args.text_final_dilate + 1),
        )
        seed_mask = cv2.dilate(seed_mask, k)

    return seed_mask


# =============================================================================
# Text edge detection
# =============================================================================

def make_text_edge_mask_from_instance(
    img_float: np.ndarray,
    inst_mask: np.ndarray,
    radius: int,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    在文字 seed instance 附近做 Canny + Scharr 边缘检测。

    inst_mask:
        只作为位置 seed。

    返回：
        edge_mask，后续会直接高斯平滑成 alpha。
    """
    h, w = img_float.shape[:2]

    inst_mask = (inst_mask > 0).astype(np.uint8) * 255

    if inst_mask.max() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    pad = max(abs(radius) * 6 + 8, 16)

    bbox = bbox_from_mask(inst_mask, pad=pad, image_shape=(h, w))
    if bbox is None:
        return np.zeros((h, w), dtype=np.uint8)

    x0, y0, x1, y1 = bbox

    luma = bgr_float_to_luma01(img_float)
    gray = np.round(luma * 255.0).clip(0, 255).astype(np.uint8)

    gray_roi = gray[y0:y1, x0:x1]
    seed_roi = inst_mask[y0:y1, x0:x1] > 0

    if seed_roi.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Canny 前轻微预平滑，降低边缘毛刺。
    # 注意这是检测用，不是模糊最终图像。
    if args.edge_preblur_sigma > 0:
        gray_blur = cv2.GaussianBlur(
            gray_roi,
            ksize=(0, 0),
            sigmaX=float(args.edge_preblur_sigma),
            sigmaY=float(args.edge_preblur_sigma),
            borderType=cv2.BORDER_REPLICATE,
        )
    else:
        gray_blur = gray_roi

    # 1. Canny
    edges_canny = cv2.Canny(
        gray_blur,
        threshold1=int(args.edge_canny_low),
        threshold2=int(args.edge_canny_high),
        L2gradient=True,
    )

    # 2. Scharr 梯度约束
    gx = cv2.Scharr(gray_blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    # 用固定参考值归一化，避免低梯度区域噪声被放大
    _grad_ref = max(float(mag.max()), 128.0)
    mag_u8 = np.round(mag / _grad_ref * 255.0).clip(0, 255).astype(np.uint8)

    # 用 seed 附近区域估计梯度阈值
    stat_gate_radius = max(abs(radius) * 3 + 3, 7)

    stat_gate = cv2.dilate(
        seed_roi.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * stat_gate_radius + 1, 2 * stat_gate_radius + 1),
        ),
    ) > 0

    stat_vals = mag_u8[stat_gate]

    if stat_vals.size > 0:
        grad_thr = max(
            int(args.edge_grad_min),
            int(np.percentile(stat_vals, args.edge_grad_percentile)),
        )
    else:
        grad_thr = int(args.edge_grad_min)

    edges_grad = mag_u8 >= grad_thr

    edges = np.logical_and(edges_canny > 0, edges_grad)

    # 3. seed gate 只限制寻找范围，不参与最终绘制
    gate_radius = int(args.edge_gate_radius)
    if gate_radius <= 0:
        gate_radius = 8

    k_gate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * gate_radius + 1, 2 * gate_radius + 1),
    )

    seed_gate = cv2.dilate(seed_roi.astype(np.uint8), k_gate) > 0
    edges = np.logical_and(edges, seed_gate)

    # 4. 连通域过滤
    edge_u8 = edges.astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_u8,
        connectivity=8,
    )

    keep = np.zeros_like(edge_u8, dtype=np.uint8)

    seed_gate_small = cv2.dilate(
        seed_roi.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0

    roi_area = edge_u8.shape[0] * edge_u8.shape[1]

    for i in range(1, num):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < args.edge_min_area:
            continue

        if area > roi_area * args.edge_max_area_ratio:
            continue

        # 必须和 seed 附近有交集
        if not np.logical_and(comp, seed_gate_small).any():
            continue

        keep[comp] = 255

    # 如果太严格导致为空，可以放宽一次
    if keep.max() == 0 and args.edge_relax_if_empty:
        relaxed = np.logical_and(edges_canny > 0, seed_gate).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            relaxed,
            connectivity=8,
        )

        for i in range(1, num):
            comp = labels == i
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < args.edge_min_area:
                continue

            if area > roi_area * args.edge_max_area_ratio:
                continue

            if not np.logical_and(comp, seed_gate_small).any():
                continue

            keep[comp] = 255

    # edge_close=1 表示不做闭合
    if args.edge_close > 1:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (args.edge_close, args.edge_close),
        )
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k_close)

    # edge_dilate=0 表示不膨胀边缘
    if args.edge_dilate > 0:
        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.edge_dilate + 1, 2 * args.edge_dilate + 1),
        )
        keep = cv2.dilate(keep, k_dilate)

    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[y0:y1, x0:x1] = keep

    return edge_mask


# =============================================================================
# Gaussian alpha / blend
# =============================================================================

def make_edge_alpha_gaussian(
    edge_mask: np.ndarray,
    base_alpha: float,
    aa_sigma: float = 0.65,
    aa_threshold: float = 0.02,
    aa_core_alpha: float = 0.85,
    aa_gamma: float = 1.0,
) -> np.ndarray:
    """
    直接对 edge_mask 做高斯平滑，生成抗锯齿 alpha。

    重点：
        1. 不做形态学膨胀；
        2. 不做 distance transform；
        3. 不模糊原图；
        4. 只平滑 edge alpha。

    edge_mask:
        0/255 二值边缘。

    aa_sigma:
        高斯平滑强度。
        0.45 较锐，0.65 默认，0.85 更平滑但略宽。

    aa_threshold:
        去掉很弱的远端 alpha，避免边缘发灰。

    aa_core_alpha:
        原始 edge 像素的最低 alpha，保证边缘核心不太淡。
    """
    edge = (edge_mask > 0).astype(np.float32)

    if edge.sum() == 0:
        return np.zeros(edge_mask.shape, dtype=np.float32)

    if aa_sigma > 0:
        alpha = cv2.GaussianBlur(
            edge,
            ksize=(0, 0),
            sigmaX=float(aa_sigma),
            sigmaY=float(aa_sigma),
            borderType=cv2.BORDER_REPLICATE,
        )
    else:
        alpha = edge.copy()

    maxv = float(alpha.max())
    if maxv > 1e-6:
        alpha = alpha / maxv

    # 去掉很弱的远处残留，避免边缘周围发灰
    if aa_threshold > 0:
        alpha[alpha < float(aa_threshold)] = 0.0

    # 保证原始边缘核心像素足够强
    alpha[edge > 0] = np.maximum(alpha[edge > 0], float(aa_core_alpha))

    # gamma 调整：
    # gamma > 1 更锐；
    # gamma < 1 更柔。
    if abs(float(aa_gamma) - 1.0) > 1e-6:
        alpha = np.power(np.clip(alpha, 0.0, 1.0), float(aa_gamma))

    alpha = alpha.astype(np.float32) * float(base_alpha)
    return np.clip(alpha, 0.0, 1.0)


def alpha_blend_color(
    img_float: np.ndarray,
    color_bgr:np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    a = alpha[:, :, None].astype(np.float32)
    c = color_bgr.reshape(1, 1, 3).astype(np.float32)

    out = img_float * (1.0 - a) + c * a
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Stroke application
# =============================================================================

def apply_text_edge_stroke(
    img_float: np.ndarray,
    text_instances: List[np.ndarray],
    radius: int,
    alpha_value: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对文字实例做基于 edge mask 高斯平滑 alpha 的描边。
    """
    out = img_float.copy()
    h, w = img_float.shape[:2]

    total_edge_mask = np.zeros((h, w), dtype=np.uint8)
    total_stroke_mask = np.zeros((h, w), dtype=np.uint8)
    total_alpha = np.zeros((h, w), dtype=np.float32)

    for inst in text_instances:
        if inst is None or inst.max() == 0:
            continue

        inst = (inst > 0).astype(np.uint8) * 255

        if int((inst > 0).sum()) < args.text_group_min_area:
            continue

        edge_mask = make_text_edge_mask_from_instance(
            img_float=out,
            inst_mask=inst,
            radius=radius,
            args=args,
        )

        if edge_mask.max() == 0:
            continue

        total_edge_mask[edge_mask > 0] = 255

        # 前景颜色：用文字 seed
        fg_color = median_color_bgr_float(
            out,
            inst,
            default=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )

        # 背景颜色：用 seed 外部环带
        context_ring = make_outer_context_ring(
            inst,
            inner_radius=max(1, radius + 1),
            outer_radius=max(radius * 4, radius + 6),
        )

        bg_color = median_color_bgr_float(
            out,
            context_ring,
            default=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )

        hsv_fg = bgr_float_to_hsv_cv(fg_color)
        hsv_bg = bgr_float_to_hsv_cv(bg_color)

        if hsv_fg[1] >= args.complement_ref_sat_thresh:
            ref_color = fg_color
        elif hsv_bg[1] >= args.complement_ref_sat_thresh:
            ref_color = bg_color
        else:
            ref_color = bg_color

        stroke_color = choose_stroke_color(
            fg_bgr=fg_color,
            bg_bgr=bg_color,
            prefer_ref_bgr=ref_color,
            min_bg_contrast=args.min_bg_contrast,
            min_fg_contrast=args.min_fg_contrast,
            min_sat=args.min_complement_sat,
        )

        alpha = make_edge_alpha_gaussian(
            edge_mask=edge_mask,
            base_alpha=alpha_value,
            aa_sigma=args.aa_sigma,
            aa_threshold=args.aa_threshold,
            aa_core_alpha=args.aa_core_alpha,
            aa_gamma=args.aa_gamma,
        )

        if alpha.max() <= 0:
            continue

        out = alpha_blend_color(out, stroke_color, alpha)

        total_stroke_mask[alpha > 0.02] = 255
        total_alpha = np.maximum(total_alpha, alpha)

    alpha_mask = np.round(total_alpha * 255.0).clip(0, 255).astype(np.uint8)

    return out, total_stroke_mask, total_edge_mask, alpha_mask


def enhance_text_edge_gaussian(
    img: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bgr, alpha = split_bgr_alpha(img)
    input_dtype = bgr.dtype

    bgr_u8 = to_uint8_for_detection(bgr)
    bgr_float = to_float01(bgr)

    text_seed_mask = detect_text_seed_mask_traditional(bgr_u8, args)

    text_instances = group_mask_instances(
        text_seed_mask,
        group_kernel_w=args.text_group_kernel_w,
        group_kernel_h=args.text_group_kernel_h,
        min_group_area=args.text_group_min_area,
    )

    out_float = bgr_float.copy()

    out_float, text_stroke_mask, text_edge_mask, alpha_mask = apply_text_edge_stroke(
        out_float,
        text_instances,
        radius=args.stroke_radius,
        alpha_value=args.stroke_alpha,
        args=args,
    )

    bgr_out = restore_dtype(out_float, input_dtype)
    out = merge_bgr_alpha(bgr_out, alpha)

    debug: Dict[str, Any] = {
        "text_seed_mask": text_seed_mask,
        "text_edge_mask": text_edge_mask,
        "alpha_mask": alpha_mask,
        "text_stroke_mask": text_stroke_mask,
        "num_text_instances": len(text_instances),
    }

    return out, debug


# =============================================================================
# Debug
# =============================================================================

def save_debug_images(
    output_path: Path,
    bgr_u8: np.ndarray,
    debug: Dict[str, Any],
) -> None:
    debug_dir = output_path.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem

    text_seed_mask = debug["text_seed_mask"]
    text_edge_mask = debug["text_edge_mask"]
    alpha_mask = debug["alpha_mask"]
    text_stroke_mask = debug["text_stroke_mask"]

    imwrite_unicode(str(debug_dir / f"{stem}_01_text_seed_mask.png"), text_seed_mask)
    imwrite_unicode(str(debug_dir / f"{stem}_02_text_edge_mask.png"), text_edge_mask)
    imwrite_unicode(str(debug_dir / f"{stem}_03_gaussian_alpha.png"), alpha_mask)
    imwrite_unicode(str(debug_dir / f"{stem}_04_text_stroke_mask.png"), text_stroke_mask)

    overlay = bgr_u8.copy()

    # seed: green
    m = text_seed_mask > 0
    overlay[m] = (
        0.65 * overlay[m].astype(np.float32)
        + 0.35 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    # raw edge: cyan
    m = text_edge_mask > 0
    overlay[m] = (
        0.35 * overlay[m].astype(np.float32)
        + 0.65 * np.array([255, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    # final alpha/stroke region: red
    m = text_stroke_mask > 0
    overlay[m] = (
        0.45 * overlay[m].astype(np.float32)
        + 0.55 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)

    imwrite_unicode(str(debug_dir / f"{stem}_05_overlay.png"), overlay)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text edge outline with direct Gaussian-smoothed alpha."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input image or folder.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process folder.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug masks.",
    )

    # 最终绘制参数
    parser.add_argument(
        "--stroke-radius",
        type=int,
        default=0,
        help=(
            "Kept for compatibility. "
            "Default 0 means no distance expansion. "
            "This script uses Gaussian alpha directly."
        ),
    )
    parser.add_argument(
        "--stroke-alpha",
        type=float,
        default=0.9,
        help="Final edge alpha scale. Default: 0.9.",
    )

    # Gaussian alpha 参数
    parser.add_argument(
        "--aa-sigma",
        type=float,
        default=0.65,
        help="Gaussian sigma for smoothing edge alpha. Default: 0.65.",
    )
    parser.add_argument(
        "--aa-threshold",
        type=float,
        default=0.02,
        help="Remove weak Gaussian alpha below this value. Default: 0.02.",
    )
    parser.add_argument(
        "--aa-core-alpha",
        type=float,
        default=0.85,
        help="Minimum alpha on original edge pixels before stroke-alpha scale. Default: 0.85.",
    )
    parser.add_argument(
        "--aa-gamma",
        type=float,
        default=1.0,
        help="Gamma for Gaussian alpha. >1 sharper, <1 softer. Default: 1.0.",
    )

    # 颜色参数
    parser.add_argument(
        "--min-bg-contrast",
        type=float,
        default=3.0,
        help="Minimum contrast between edge color and local background. Default: 3.0.",
    )
    parser.add_argument(
        "--min-fg-contrast",
        type=float,
        default=1.1,
        help="Minimum contrast between edge color and foreground. Default: 1.1.",
    )
    parser.add_argument(
        "--min-complement-sat",
        type=int,
        default=150,
        help="Minimum saturation for complementary color candidates. Default: 150.",
    )
    parser.add_argument(
        "--complement-ref-sat-thresh",
        type=int,
        default=35,
        help="Use foreground hue if saturation >= this threshold. Default: 35.",
    )

    # 文字 seed 检测参数
    parser.add_argument(
        "--use-clahe",
        action="store_true",
        help="Use CLAHE before seed detection.",
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="CLAHE clip limit. Default: 2.0.",
    )
    parser.add_argument(
        "--use-mser",
        dest="use_mser",
        action="store_true",
        default=False,
        help="Enable MSER for text seed. Default: disabled.",
    )
    parser.add_argument(
        "--disable-mser",
        dest="use_mser",
        action="store_false",
        help="Disable MSER (default behavior).",
    )

    parser.add_argument(
        "--text-local-sigma",
        type=float,
        default=7.0,
        help="Gaussian sigma for local background. Default: 7.",
    )
    parser.add_argument(
        "--text-diff-thresh",
        type=int,
        default=15,
        help="Local contrast threshold for seed. Default: 15.",
    )
    parser.add_argument(
        "--text-grad-thresh",
        type=int,
        default=30,
        help="Gradient threshold for seed. Default: 30.",
    )
    parser.add_argument(
        "--text-morph-kernel",
        type=int,
        default=25,
        help="Morph kernel for top-hat/black-hat. Default: 25.",
    )
    parser.add_argument(
        "--text-response-thresh",
        type=float,
        default=12.0,
        help="Minimum response threshold. Default: 12.",
    )
    parser.add_argument(
        "--text-response-std-scale",
        type=float,
        default=0.8,
        help="Dynamic response threshold scale. Default: 0.8.",
    )

    parser.add_argument(
        "--text-min-area",
        type=int,
        default=50,
        help="Minimum seed component area. Default: 50.",
    )
    parser.add_argument(
        "--text-max-area",
        type=int,
        default=3000,
        help="Maximum seed component area. Default: 3000.",
    )
    parser.add_argument(
        "--text-min-width",
        type=int,
        default=6,
        help="Minimum seed width. Default: 6.",
    )
    parser.add_argument(
        "--text-min-height",
        type=int,
        default=12,
        help="Minimum seed height. Default: 12.",
    )
    parser.add_argument(
        "--text-max-height",
        type=int,
        default=60,
        help="Maximum seed height. Default: 60.",
    )
    parser.add_argument(
        "--text-min-fill",
        type=float,
        default=0.05,
        help="Minimum seed fill ratio. Default: 0.05.",
    )
    parser.add_argument(
        "--text-max-fill",
        type=float,
        default=0.75,
        help="Maximum seed fill ratio. Default: 0.75.",
    )
    parser.add_argument(
        "--text-max-aspect",
        type=float,
        default=12.0,
        help="Maximum seed aspect ratio. Default: 12.",
    )
    parser.add_argument(
        "--text-cleanup-open",
        type=int,
        default=0,
        help="Opening kernel size for seed cleanup. 0 disables. Default: 0.",
    )
    parser.add_argument(
        "--text-cleanup-close",
        type=int,
        default=0,
        help="Closing kernel size for seed cleanup. 0 disables. Default: 0.",
    )
    parser.add_argument(
        "--text-final-dilate",
        type=int,
        default=0,
        help="Final dilation for seed. Default: 0.",
    )

    # 文字边缘检测参数
    parser.add_argument(
        "--edge-preblur-sigma",
        type=float,
        default=0.8,
        help="Gaussian pre-blur sigma before Canny. Only for detection. Default: 0.8.",
    )
    parser.add_argument(
        "--edge-canny-low",
        type=int,
        default=40,
        help="Canny low threshold for text edge. Default: 40.",
    )
    parser.add_argument(
        "--edge-canny-high",
        type=int,
        default=120,
        help="Canny high threshold for text edge. Default: 120.",
    )
    parser.add_argument(
        "--edge-grad-min",
        type=int,
        default=18,
        help="Minimum Scharr gradient threshold. Default: 18.",
    )
    parser.add_argument(
        "--edge-grad-percentile",
        type=float,
        default=55.0,
        help="Gradient percentile near seed. Default: 55.",
    )
    parser.add_argument(
        "--edge-gate-radius",
        type=int,
        default=5,
        help=(
            "Seed dilation radius only for searching/gating edges. "
            "It is not drawn. Default: 5."
        ),
    )
    parser.add_argument(
        "--edge-min-area",
        type=int,
        default=8,
        help="Minimum connected edge component area. Default: 8.",
    )
    parser.add_argument(
        "--edge-max-area-ratio",
        type=float,
        default=0.15,
        help="Maximum edge component area ratio in ROI. Default: 0.15.",
    )
    parser.add_argument(
        "--edge-close",
        type=int,
        default=1,
        help="Close kernel size for edge mask. 1 disables. Default: 1.",
    )
    parser.add_argument(
        "--edge-dilate",
        type=int,
        default=0,
        help="Optional dilation for edge mask. 0 disables. Default: 0.",
    )
    parser.add_argument(
        "--edge-relax-if-empty",
        action="store_true",
        default=False,
        help="Relax edge filter if no edge is found. Default: disabled.",
    )

    # 文字分组参数
    parser.add_argument(
        "--text-group-kernel-w",
        type=int,
        default=30,
        help="Text grouping horizontal kernel. Default: 30.",
    )
    parser.add_argument(
        "--text-group-kernel-h",
        type=int,
        default=8,
        help="Text grouping vertical kernel. Default: 8.",
    )
    parser.add_argument(
        "--text-group-min-area",
        type=int,
        default=200,
        help="Minimum grouped text seed area. Default: 200.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = collect_images(str(input_path), recursive=args.recursive)

    if not paths:
        print(f"[Error] No images found: {input_path}")
        return

    print("=" * 80)
    print("Text Edge Gaussian-Smoothed Outline")
    print("=" * 80)
    print(f"Input              : {input_path}")
    print(f"Output             : {output_root}")
    print(f"Images             : {len(paths)}")
    print(f"Stroke radius      : {args.stroke_radius}")
    print(f"Stroke alpha       : {args.stroke_alpha}")
    print(f"AA sigma           : {args.aa_sigma}")
    print(f"AA threshold       : {args.aa_threshold}")
    print(f"AA core alpha      : {args.aa_core_alpha}")
    print(f"Edge preblur sigma : {args.edge_preblur_sigma}")
    print(f"Edge dilate        : {args.edge_dilate}")
    print(f"Edge close         : {args.edge_close}")
    print(f"Save debug         : {args.save_debug}")
    print("=" * 80)

    for idx, p in enumerate(paths, 1):
        print(f"[{idx}/{len(paths)}] {p}")

        img = imread_unicode(str(p))

        if img is None:
            print(f"  [Warning] Failed to read: {p}")
            continue

        try:
            out, debug = enhance_text_edge_gaussian(img, args)
        except Exception as e:
            print(f"  [Warning] Failed to process: {p}, error={e}")
            continue

        out_path = get_relative_output_path(
            image_path=p,
            input_root=input_path,
            output_root=output_root,
            suffix="_edge_gaussian",
        )

        ok =imwrite_unicode(str(out_path), out)

        if not ok:
            print(f"  [Warning] Failed to save: {out_path}")
            continue

        print(
            f"  Saved: {out_path} | "
            f"text_instances={debug['num_text_instances']}"
        )

        if args.save_debug:
            bgr, _ = split_bgr_alpha(img)
            bgr_u8 = to_uint8_for_detection(bgr)
            save_debug_images(out_path, bgr_u8, debug)

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()