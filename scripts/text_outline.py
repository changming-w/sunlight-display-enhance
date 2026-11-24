#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_stroke_outline.py — 沿检测到的边缘轮廓绘制高反差互补色描边

特点:
    1. 不再对 edge mask 直接膨胀填色，避免碎点/噪声感
    2. 使用 Lab 多通道 Canny 检测文字/图标/线条边缘
    3. 清理边缘小碎点，并连接轻微断裂边缘
    4. 使用 findContours 提取连续边缘轮廓
    5. 每条轮廓统一一种颜色，避免逐像素互补色导致彩色噪声
    6. 使用 cv2.drawContours + LINE_AA 沿轮廓画线，线条更连续

Usage:
    python edge_stroke_outline.py --input test/ui.png --output test/out --save-masks

推荐参数:
    python edge_stroke_outline.py \
        --input test/ui.png \
        --output test/out \
        --edge-percentile 88 \
        --min-high 45 \
        --close-iter 1 \
        --min-edge-area 8 \
        --min-contour-len 18 \
        --thickness 1 \
        --blend 0.95 \
        --color-mode complement \
        --save-masks
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Unicode-safe IO
# ============================================================

def imread_unicode(path: str):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    path = str(path)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = out_path.suffix
    if not ext:
        ext = ".png"

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False

    buf.tofile(path)
    return True


def normalize_input_image(img: np.ndarray):
    if img is None:
        return None

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    return []


# ============================================================
# Basic utilities
# ============================================================

def make_odd(v: int, min_value: int = 3) -> int:
    v = int(v)
    if v < min_value:
        v = min_value
    if v % 2 == 0:
        v += 1
    return v


def remove_small_components(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    去除二值 mask 中的小连通域。
    """
    mask = (mask > 0).astype(np.uint8) * 255

    if min_area <= 1:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    out = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out


# ============================================================
# Edge detection
# ============================================================

def canny_by_gradient_percentile(
    ch: np.ndarray,
    edge_percentile: float = 88.0,
    low_ratio: float = 0.40,
    min_high: float = 45.0,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    基于 Scharr 梯度分位数自动确定 Canny 阈值。

    相比用图像灰度 median 做 auto-canny，这个更适合 UI/文字边缘。
    """
    if blur_ksize and blur_ksize >= 3:
        blur_ksize = make_odd(blur_ksize, 3)
        ch_blur = cv2.GaussianBlur(ch, (blur_ksize, blur_ksize), 0)
    else:
        ch_blur = ch

    ch_f = ch_blur.astype(np.float32)

    gx = cv2.Scharr(ch_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(ch_f, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)

    nz = mag[mag > 1e-3]

    if len(nz) == 0:
        return np.zeros_like(ch, dtype=np.uint8)

    high = max(float(np.percentile(nz, edge_percentile)), float(min_high))
    low = max(0.0, high * float(low_ratio))

    edges = cv2.Canny(
        ch_blur,
        threshold1=low,
        threshold2=high,
        L2gradient=True
    )

    return edges


def detect_lab_canny_edges(
    img_bgr: np.ndarray,
    edge_percentile: float = 88.0,
    low_ratio: float = 0.40,
    min_high: float = 45.0,
    chroma_enable: bool = True,
    chroma_min_std: float = 2.0,
    chroma_min_high_scale: float = 0.65,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Lab 多通道 Canny 边缘检测。

    L:
        亮度边缘，负责大部分文字/图标边界。

    a/b:
        色度边缘，负责等亮度但不同颜色的 UI 边界。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    edge_L = canny_by_gradient_percentile(
        L,
        edge_percentile=edge_percentile,
        low_ratio=low_ratio,
        min_high=min_high,
        blur_ksize=blur_ksize
    )

    edges = edge_L.copy()

    if chroma_enable:
        chroma_min_high = max(8.0, min_high * chroma_min_high_scale)

        if np.std(A) >= chroma_min_std:
            edge_A = canny_by_gradient_percentile(
                A,
                edge_percentile=edge_percentile,
                low_ratio=low_ratio,
                min_high=chroma_min_high,
                blur_ksize=blur_ksize
            )
            edges = cv2.bitwise_or(edges, edge_A)

        if np.std(B) >= chroma_min_std:
            edge_B = canny_by_gradient_percentile(
                B,
                edge_percentile=edge_percentile,
                low_ratio=low_ratio,
                min_high=chroma_min_high,
                blur_ksize=blur_ksize
            )
            edges = cv2.bitwise_or(edges, edge_B)

    return edges


def clean_edge_mask(
    edge_mask: np.ndarray,
    close_iter: int = 1,
    open_iter: int = 0,
    min_edge_area: int = 8,
) -> np.ndarray:
    """
    清理边缘 mask:
        1. close: 连接轻微断裂边缘
        2. open: 去除很小毛刺，默认不开，避免断字
        3. 连通域面积过滤
    """
    edge = (edge_mask > 0).astype(np.uint8) * 255

    if close_iter > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge = cv2.morphologyEx(
            edge,
            cv2.MORPH_CLOSE,
            k_close,
            iterations=int(close_iter)
        )

    if open_iter > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edge = cv2.morphologyEx(
            edge,
            cv2.MORPH_OPEN,
            k_open,
            iterations=int(open_iter)
        )

    edge = remove_small_components(edge, min_area=min_edge_area)

    return edge


# ============================================================
# Color utilities
# ============================================================

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4
    )


def relative_luminance_bgr_color(bgr: np.ndarray) -> float:
    """
    单个 BGR 颜色的 WCAG 相对亮度。
    """
    arr = bgr.astype(np.float32) / 255.0
    rgb = arr[::-1]
    lin = srgb_to_linear(rgb)

    r, g, b = lin[0], lin[1], lin[2]

    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio_bgr(c1: np.ndarray, c2: np.ndarray) -> float:
    y1 = relative_luminance_bgr_color(c1)
    y2 = relative_luminance_bgr_color(c2)

    hi = max(y1, y2)
    lo = min(y1, y2)

    return float((hi + 0.05) / (lo + 0.05))


def median_color_from_mask(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    fallback=None
) -> np.ndarray:
    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        if fallback is None:
            return np.array([128, 128, 128], dtype=np.uint8)
        return fallback.astype(np.uint8)

    pixels = img_bgr[ys, xs]
    return np.median(pixels, axis=0).astype(np.uint8)


def choose_high_contrast_complement_color(
    base_bgr: np.ndarray,
    color_mode: str = "complement",
) -> tuple:
    """
    为一条轮廓选择统一描边颜色。

    color_mode:
        complement:
            有彩色区域用互补 hue，并搜索最合适亮度；
            黑白灰区域用黑/白最大反差。

        bw:
            只用黑/白高反差。

        invert:
            直接反相色。

        black / white:
            固定黑/白。
    """
    base_bgr = base_bgr.astype(np.uint8)

    if color_mode == "black":
        return (0, 0, 0)

    if color_mode == "white":
        return (255, 255, 255)

    if color_mode == "invert":
        c = 255 - base_bgr
        return (int(c[0]), int(c[1]), int(c[2]))

    y_simple = 0.114 * base_bgr[0] + 0.587 * base_bgr[1] + 0.299 * base_bgr[2]

    if color_mode == "bw":
        if y_simple > 128:
            return (0, 0, 0)
        else:
            return (255, 255, 255)

    # complement mode
    hsv = cv2.cvtColor(
        base_bgr.reshape(1, 1, 3),
        cv2.COLOR_BGR2HSV
    )[0, 0]

    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 黑白灰没有稳定互补 hue，直接用黑白最大反差
    if s < 35:
        if y_simple > 128:
            return (0, 0, 0)
        else:
            return (255, 255, 255)

    comp_h = (h + 90) % 180

    sat_candidates = [255, 235, 210]
    val_candidates = [24, 36, 48, 64, 80, 96, 128, 160, 192, 224, 255]

    best_color = None
    best_score = -1.0

    for ss in sat_candidates:
        for vv in val_candidates:
            hsv_c = np.array([[[comp_h, ss, vv]]], dtype=np.uint8)
            bgr_c = cv2.cvtColor(hsv_c, cv2.COLOR_HSV2BGR)[0, 0]

            cr = contrast_ratio_bgr(bgr_c, base_bgr)

            # 偏好高对比，同时稍微偏好高饱和
            score = cr + 0.03 * (ss / 255.0)

            if score > best_score:
                best_score = score
                best_color = bgr_c

    return (int(best_color[0]), int(best_color[1]), int(best_color[2]))


# ============================================================
# Contour extraction and drawing
# ============================================================

def extract_edge_contours(
    edge_mask: np.ndarray,
    min_contour_len: float = 18.0,
    max_contour_len_ratio: float = 2.5,
    simplify_epsilon: float = 0.0,
) -> list:
    """
    从边缘 mask 提取连续轮廓，并过滤短碎线。

    注意:
        这里使用 RETR_LIST，保留所有线条轮廓。
    """
    H, W = edge_mask.shape[:2]
    max_len = max_contour_len_ratio * float(H + W)

    contours, _ = cv2.findContours(
        edge_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    kept = []

    for cnt in contours:
        if cnt is None or len(cnt) < 2:
            continue

        length = cv2.arcLength(cnt, closed=False)

        if length < min_contour_len:
            continue

        if length > max_len:
            # 超大轮廓多数是大面板边框/背景边缘。
            # 如果你想保留大边框，可以把 max_contour_len_ratio 调大。
            continue

        if simplify_epsilon > 0:
            eps = float(simplify_epsilon)
            cnt = cv2.approxPolyDP(cnt, epsilon=eps, closed=False)

        kept.append(cnt)

    return kept


def choose_color_for_contour(
    img_bgr: np.ndarray,
    contour: np.ndarray,
    sample_thickness: int = 3,
    color_mode: str = "complement",
) -> tuple:
    """
    每条轮廓只取一个统一颜色，避免逐像素颜色跳变产生噪声。
    """
    H, W = img_bgr.shape[:2]

    mask = np.zeros((H, W), dtype=np.uint8)

    cv2.drawContours(
        mask,
        [contour],
        contourIdx=-1,
        color=255,
        thickness=max(1, int(sample_thickness)),
        lineType=cv2.LINE_8
    )

    base_color = median_color_from_mask(img_bgr, mask)

    return choose_high_contrast_complement_color(
        base_color,
        color_mode=color_mode
    )


def draw_contours_as_clean_lines(
    img_bgr: np.ndarray,
    contours: list,
    thickness: int = 1,
    blend_strength: float = 0.95,
    color_mode: str = "complement",
    sample_thickness: int = 3,
    anti_alias: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    沿轮廓画连续线条，而不是对 mask 做碎点填色。

    返回:
        result
        alpha_layer
        color_layer
    """
    H, W = img_bgr.shape[:2]

    color_layer = np.zeros_like(img_bgr, dtype=np.uint8)
    alpha_layer = np.zeros((H, W), dtype=np.uint8)

    line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8
    thickness = max(1, int(thickness))

    for cnt in contours:
        color = choose_color_for_contour(
            img_bgr,
            cnt,
            sample_thickness=sample_thickness,
            color_mode=color_mode
        )

        # 每条轮廓单独生成临时 alpha。
        # 这样可以避免直接画到 color_layer 时 AA 半透明像素被黑底污染。
        temp_alpha = np.zeros((H, W), dtype=np.uint8)

        cv2.drawContours(
            temp_alpha,
            [cnt],
            contourIdx=-1,
            color=255,
            thickness=thickness,
            lineType=line_type
        )

        update = temp_alpha > alpha_layer

        color_layer[update] = color
        alpha_layer = np.maximum(alpha_layer, temp_alpha)

    img_f = img_bgr.astype(np.float32)
    color_f = color_layer.astype(np.float32)

    alpha = alpha_layer.astype(np.float32) / 255.0
    alpha = np.clip(alpha * float(blend_strength), 0.0, 1.0)

    result = img_f * (1.0 - alpha[:, :, None]) + color_f * alpha[:, :, None]

    return (
        np.clip(result, 0, 255).astype(np.uint8),
        alpha_layer,
        color_layer
    )


# ============================================================
# Full process
# ============================================================

def process_image(
    img_bgr: np.ndarray,
    edge_percentile: float = 88.0,
    low_ratio: float = 0.40,
    min_high: float = 45.0,
    chroma_enable: bool = True,
    blur_ksize: int = 3,
    close_iter: int = 1,
    open_iter: int = 0,
    min_edge_area: int = 8,
    min_contour_len: float = 18.0,
    max_contour_len_ratio: float = 2.5,
    simplify_epsilon: float = 0.0,
    thickness: int = 1,
    blend_strength: float = 0.95,
    color_mode: str = "complement",
    sample_thickness: int = 3,
    anti_alias: bool = True,
):
    """
    主流程:
        Lab Canny -> clean edge -> contours -> draw clean contour lines
    """
    raw_edges = detect_lab_canny_edges(
        img_bgr,
        edge_percentile=edge_percentile,
        low_ratio=low_ratio,
        min_high=min_high,
        chroma_enable=chroma_enable,
        blur_ksize=blur_ksize
    )

    clean_edges = clean_edge_mask(
        raw_edges,
        close_iter=close_iter,
        open_iter=open_iter,
        min_edge_area=min_edge_area
    )

    contours = extract_edge_contours(
        clean_edges,
        min_contour_len=min_contour_len,
        max_contour_len_ratio=max_contour_len_ratio,
        simplify_epsilon=simplify_epsilon
    )

    result, alpha_layer, color_layer = draw_contours_as_clean_lines(
        img_bgr,
        contours,
        thickness=thickness,
        blend_strength=blend_strength,
        color_mode=color_mode,
        sample_thickness=sample_thickness,
        anti_alias=anti_alias
    )

    debug = {
        "raw_edges": raw_edges,
        "clean_edges": clean_edges,
        "alpha_layer": alpha_layer,
        "color_layer": color_layer,
        "contour_count": len(contours),
    }

    return result, debug


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="沿图像边缘轮廓绘制连续高反差互补色描边"
    )

    parser.add_argument("--input", type=str, required=True,
                        help="输入图片文件或目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")

    # Edge detection
    parser.add_argument("--edge-percentile", type=float, default=88.0,
                        help="边缘梯度分位数阈值，越大边缘越少")
    parser.add_argument("--low-ratio", type=float, default=0.40,
                        help="Canny 低阈值 = 高阈值 * low_ratio")
    parser.add_argument("--min-high", type=float, default=45.0,
                        help="Canny 高阈值下限")
    parser.add_argument("--no-chroma", action="store_true",
                        help="关闭 Lab 色度边缘，只用亮度边缘")
    parser.add_argument("--blur-ksize", type=int, default=3,
                        help="Canny 前高斯模糊核大小，0 表示不模糊")

    # Edge clean
    parser.add_argument("--close-iter", type=int, default=1,
                        help="闭运算次数，用于连接断裂边缘")
    parser.add_argument("--open-iter", type=int, default=0,
                        help="开运算次数，用于去毛刺，通常不建议太大")
    parser.add_argument("--min-edge-area", type=int, default=8,
                        help="边缘连通域最小面积，小于该值视为噪声")

    # Contour filter
    parser.add_argument("--min-contour-len", type=float, default=18.0,
                        help="最小轮廓长度，越大越能去碎点")
    parser.add_argument("--max-contour-len-ratio", type=float, default=2.5,
                        help="最大轮廓长度比例，过大轮廓会被过滤")
    parser.add_argument("--simplify-epsilon", type=float, default=0.0,
                        help="轮廓简化 epsilon，0 表示不简化")

    # Drawing
    parser.add_argument("--thickness", type=int, default=1,
                        help="线条描边粗细")
    parser.add_argument("--blend", type=float, default=0.95,
                        help="描边混合强度 0~1")
    parser.add_argument("--color-mode", type=str, default="complement",
                        choices=["complement", "bw", "invert", "black", "white"],
                        help="描边颜色模式")
    parser.add_argument("--sample-thickness", type=int, default=3,
                        help="为轮廓估计颜色时的采样线宽")
    parser.add_argument("--no-aa", action="store_true",
                        help="关闭抗锯齿线条")

    parser.add_argument("--save-masks", action="store_true",
                        help="保存 raw_edges / clean_edges / alpha / color 调试图")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_path)

    if not image_paths:
        print(f"Input not found or no images: {input_path}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    for img_path in image_paths:
        img = imread_unicode(str(img_path))
        img_bgr = normalize_input_image(img)

        if img_bgr is None:
            print(f"  SKIP: {img_path.name}")
            continue

        result, debug = process_image(
            img_bgr,
            edge_percentile=args.edge_percentile,
            low_ratio=args.low_ratio,
            min_high=args.min_high,
            chroma_enable=not args.no_chroma,
            blur_ksize=args.blur_ksize,
            close_iter=args.close_iter,
            open_iter=args.open_iter,
            min_edge_area=args.min_edge_area,
            min_contour_len=args.min_contour_len,
            max_contour_len_ratio=args.max_contour_len_ratio,
            simplify_epsilon=args.simplify_epsilon,
            thickness=args.thickness,
            blend_strength=args.blend,
            color_mode=args.color_mode,
            sample_thickness=args.sample_thickness,
            anti_alias=not args.no_aa
        )

        out_path = output_dir / img_path.name
        imwrite_unicode(str(out_path), result)

        print(
            f"  {img_path.name:40s} → {out_path.name} "
            f"(contours={debug['contour_count']})"
        )

        if args.save_masks:
            stem = img_path.stem

            imwrite_unicode(
                str(output_dir / f"{stem}_raw_edges.png"),
                debug["raw_edges"]
            )

            imwrite_unicode(
                str(output_dir / f"{stem}_clean_edges.png"),
                debug["clean_edges"]
            )

            imwrite_unicode(
                str(output_dir / f"{stem}_alpha_layer.png"),
                debug["alpha_layer"]
            )

            color_vis = debug["color_layer"].copy()
            color_vis[debug["alpha_layer"] == 0] = [255, 255, 255]

            imwrite_unicode(
                str(output_dir / f"{stem}_color_layer.png"),
                color_vis
            )

    print(f"\nDone. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()