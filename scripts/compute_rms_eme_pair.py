#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Local RMS Contrast, EME and Raw Contrast for original/enhanced image folders.

只使用正方形 block：--block-size。

Outputs:
    original_metrics.csv
    enhanced_metrics.csv
    paired_comparison_metrics.csv
    pair_metric_figures/*.png  optional

Metrics:
    rms_*:
        全图 local RMS contrast 统计。

    edge_rms_*:
        仅在边缘像素位置统计 local RMS contrast。

    eme_*:
        全图 block-wise EME，单位 dB。

    edge_eme_*:
        先筛选“含边缘像素的 block”，再在完整 block 上计算 EME，单位 dB。

    raw_contrast_*:
        EME log 前的 ratio，即 (high + eps) / (low + eps)。

    edge_raw_contrast_*:
        先筛选“含边缘像素的 block”，再在完整 block 上计算 raw contrast ratio。

Usage:
    python scripts/compute_rms_eme_pair.py \
        --original ./test/simulate_sunlight/input_80000lux/ \
        --enhanced ./test/simulate_sunlight/enhanced_80000lux/ \
        --output ./test/simulate_sunlight/rms_eme_eval_80000lux \
        --rms-window 31 \
        --rms-filter gaussian \
        --rms-valid-min-mean 0.05 \
        --eme-mode classic \
        --block-size 32 \
        --edge-canny-low 0.25 \
        --edge-canny-high 0.50 \
        --edge-min-block-ratio 0.05 \
        --save-edge-maps \
        --save-edge-overlays \
        --save-pair-metric-figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# =============================================================================
# IO
# =============================================================================

def imread_unicode(path: str) -> np.ndarray | None:
    """支持中文路径的 OpenCV 读图。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """支持中文路径的 OpenCV 写图。"""
    try:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ext = out_path.suffix
        if ext == "":
            ext = ".png"

        ok, encoded = cv2.imencode(ext, img)
        if not ok:
            return False

        encoded.tofile(str(out_path))
        return True

    except Exception:
        return False

def collect_image_paths(input_path: str, recursive: bool = False) -> list[Path]:
    p = Path(input_path)

    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []

    if not p.exists() or not p.is_dir():
        return []

    iterator = p.rglob("*") if recursive else p.glob("*")

    return sorted([
        x for x in iterator
        if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    ])


def build_image_dict(
    folder: str,
    recursive: bool = False,
    match_mode: str = "basename"
) -> tuple[dict[str, Path], list[str]]:
    """
    match_mode:
        basename: 按文件名匹配，例如 a.png
        relative: 按相对路径匹配，例如 scene1/a.png
    """
    root = Path(folder)
    paths = collect_image_paths(folder, recursive=recursive)

    image_dict: dict[str, Path] = {}
    duplicate_keys: list[str] = []

    base_dir = root.parent if root.is_file() else root

    for p in paths:
        if match_mode == "relative":
            try:
                key = p.relative_to(base_dir).as_posix()
            except ValueError:
                key = p.name
        elif match_mode == "basename":
            key = p.name
        else:
            raise ValueError(f"Unknown match mode: {match_mode}")

        if key in image_dict:
            duplicate_keys.append(key)
            continue

        image_dict[key] = p

    return image_dict, duplicate_keys


def write_text_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item) + "\n")


def write_csv(
    csv_path: Path,
    rows: list[dict[str, Any]],
    preferred_fields: list[str] | None = None
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write("")
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames: list[str] = []

    if preferred_fields is not None:
        for k in preferred_fields:
            if k in all_keys and k not in fieldnames:
                fieldnames.append(k)

    for k in sorted(all_keys):
        if k not in fieldnames:
            fieldnames.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# Image conversion
# =============================================================================

def normalize_to_float01(img: np.ndarray) -> np.ndarray:
    """将 uint8/uint16/float 图像归一化到 float32 [0, 1]。"""
    if np.issubdtype(img.dtype, np.integer):
        maxv = np.iinfo(img.dtype).max
        out = img.astype(np.float32) / float(maxv)
    else:
        out = img.astype(np.float32)
        finite = out[np.isfinite(out)]

        if finite.size > 0:
            vmax = float(np.max(finite))
            if vmax > 1.5:
                if vmax <= 255.0:
                    out = out / 255.0
                elif vmax <= 65535.0:
                    out = out / 65535.0
                else:
                    out = out / vmax

    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)

    return out.astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)

    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


def to_luminance_float01(
    img: np.ndarray,
    luma: str = "bt601",
    linearize_srgb: bool = False
) -> np.ndarray:
    """
    转亮度通道，输出 float32 [0, 1]。

    OpenCV 读取彩色图为 BGR。

    bt601:
        Y = 0.299R + 0.587G + 0.114B

    bt709:
        Y = 0.2126R + 0.7152G + 0.0722B
    """
    if img.ndim == 2:
        y = normalize_to_float01(img)
        if linearize_srgb:
            y = srgb_to_linear(y)
        return y.astype(np.float32)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.shape[2] == 4:
        img = img[:, :, :3]

    f = normalize_to_float01(img)

    if linearize_srgb:
        f = srgb_to_linear(f)

    b = f[:, :, 0]
    g = f[:, :, 1]
    r = f[:, :, 2]

    if luma == "bt601":
        y = 0.299 * r + 0.587 * g + 0.114 * b
    elif luma == "bt709":
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    else:
        raise ValueError(f"Unknown luma mode: {luma}")

    return np.clip(y, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Local RMS contrast and edge detection
# =============================================================================

def ensure_odd_kernel(k: int) -> int:
    k = int(k)

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    return k


def get_block_size(args: argparse.Namespace) -> int:
    """统一获取正方形 block size。"""
    return max(1, int(args.block_size))


def compute_local_rms_contrast(
    y: np.ndarray,
    window: int = 15,
    filter_type: str = "gaussian",
    sigma: float = 0.0,
    eps: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    局部 RMS 对比度：

        C(x) = local_std(Y) / (local_mean(Y) + eps)

    返回：
        contrast_map, local_mean, local_std
    """
    y = y.astype(np.float32)
    k = ensure_odd_kernel(window)

    if filter_type == "gaussian":
        mean = cv2.GaussianBlur(
            y,
            (k, k),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT101
        )
        mean_sq = cv2.GaussianBlur(
            y * y,
            (k, k),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT101
        )

    elif filter_type == "box":
        mean = cv2.boxFilter(
            y,
            ddepth=-1,
            ksize=(k, k),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101
        )
        mean_sq = cv2.boxFilter(
            y * y,
            ddepth=-1,
            ksize=(k, k),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101
        )

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    var = mean_sq - mean * mean
    var = np.maximum(var, 0.0)

    std = np.sqrt(var)
    contrast = std / (mean + eps)
    contrast = np.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)

    return contrast.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def detect_edges(
    y: np.ndarray,
    method: str = "canny",
    canny_low: float = 0.05,
    canny_high: float = 0.15,
    sobel_ksize: int = 3,
    sobel_thresh: float = 0.1
) -> np.ndarray:
    """
    边缘检测，返回二值掩码 0/1。

    method:
        canny: Canny 边缘检测
        sobel: Sobel 梯度幅度阈值
    """
    y = y.astype(np.float32)

    if method == "canny":
        y8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(
            y8,
            threshold1=int(canny_low * 255),
            threshold2=int(canny_high * 255)
        )
        return (edges > 0).astype(np.uint8)

    elif method == "sobel":
        gx = cv2.Sobel(
            y,
            cv2.CV_32F,
            1,
            0,
            ksize=sobel_ksize,
            borderType=cv2.BORDER_REFLECT101
        )
        gy = cv2.Sobel(
            y,
            cv2.CV_32F,
            0,
            1,
            ksize=sobel_ksize,
            borderType=cv2.BORDER_REFLECT101
        )
        mag = np.sqrt(gx * gx + gy * gy)

        return (mag > sobel_thresh).astype(np.uint8)

    else:
        raise ValueError(f"Unknown edge detection method: {method}")


def summarize_contrast_map(
    contrast: np.ndarray,
    mask: np.ndarray | None = None
) -> dict[str, float]:
    valid = np.isfinite(contrast)

    if mask is not None:
        valid = valid & mask

    v = contrast[valid]

    if v.size == 0:
        return {
            "rms_valid_ratio": 0.0,
            "rms_mean": 0.0,
            "rms_map_std": 0.0,
            "rms_median": 0.0,
            "rms_p75": 0.0,
            "rms_p90": 0.0,
            "rms_p95": 0.0,
            "rms_max": 0.0,
        }

    return {
        "rms_valid_ratio": float(v.size / contrast.size),
        "rms_mean": float(np.mean(v)),
        "rms_map_std": float(np.std(v)),
        "rms_median": float(np.median(v)),
        "rms_p75": float(np.percentile(v, 75)),
        "rms_p90": float(np.percentile(v, 90)),
        "rms_p95": float(np.percentile(v, 95)),
        "rms_max": float(np.max(v)),
    }


def summarize_edge_rms_contrast(values: np.ndarray) -> dict[str, float]:
    """边缘像素位置的 RMS 对比度汇总。"""
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return {
            "edge_rms_mean": 0.0,
            "edge_rms_median": 0.0,
            "edge_rms_p90": 0.0,
            "edge_rms_num_pixels": 0.0,
        }

    return {
        "edge_rms_mean": float(np.mean(valid)),
        "edge_rms_median": float(np.median(valid)),
        "edge_rms_p90": float(np.percentile(valid, 90)),
        "edge_rms_num_pixels": float(valid.size),
    }


# =============================================================================
# Block contrast: EME / Raw Contrast
# =============================================================================

def get_eme_modes(eme_mode: str) -> list[str]:
    if eme_mode == "both":
        return ["classic", "percentile"]

    return [eme_mode]


def get_dashboard_eme_mode(args: argparse.Namespace) -> str:
    """
    Dashboard 只展示一种 EME mode。
    如果 --eme-mode both，则默认展示 classic。
    """
    modes = get_eme_modes(args.eme_mode)

    return "classic" if "classic" in modes else modes[0]


def iter_blocks(
    y: np.ndarray,
    block_size: int,
    drop_incomplete: bool = False
) -> Iterator[tuple[int, int, int, int, np.ndarray]]:
    h, w = y.shape
    bs = max(1, int(block_size))

    for y0 in range(0, h, bs):
        y1 = min(y0 + bs, h)

        if drop_incomplete and (y1 - y0) < bs:
            continue

        for x0 in range(0, w, bs):
            x1 = min(x0 + bs, w)

            if drop_incomplete and (x1 - x0) < bs:
                continue

            block = y[y0:y1, x0:x1]

            if block.size == 0:
                continue

            yield y0, y1, x0, x1, block


def compute_block_ratio(
    block: np.ndarray,
    mode: str = "classic",
    p_low: float = 5.0,
    p_high: float = 95.0,
    eps: float = 1.0 / 255.0
) -> float:
    if mode == "classic":
        low = float(np.min(block))
        high = float(np.max(block))

    elif mode == "percentile":
        low = float(np.percentile(block, p_low))
        high = float(np.percentile(block, p_high))

    else:
        raise ValueError(f"Unknown contrast mode: {mode}")

    if high < low:
        high, low = low, high

    ratio = (high + eps) / (low + eps)
    ratio = max(float(ratio), 1.0)

    return ratio


def compute_block_contrast_scores(
    y: np.ndarray,
    block_size: int = 16,
    mode: str = "classic",
    p_low: float = 5.0,
    p_high: float = 95.0,
    eps: float = 1.0 / 255.0,
    drop_incomplete: bool = False,
    log_db: bool = True,
    edge_mask: np.ndarray | None = None,
    min_edge_ratio: float = 0.0
) -> np.ndarray:
    """
    计算 block-wise contrast score。

    log_db=True:
        EME = 20 * log10((high + eps) / (low + eps))

    log_db=False:
        raw contrast = (high + eps) / (low + eps)

    edge_mask is not None:
        只选择“含边缘像素的 block”，但 score 仍在完整 block 上计算。
    """
    if mode not in ["classic", "percentile"]:
        raise ValueError(f"Unknown contrast mode: {mode}")

    y = y.astype(np.float32)
    scores: list[float] = []

    use_edge_filter = edge_mask is not None

    if use_edge_filter:
        edge_mask_bool = edge_mask.astype(bool)
        min_edge_ratio = max(0.0, float(min_edge_ratio))
    else:
        edge_mask_bool = None

    for y0, y1, x0, x1, block in iter_blocks(
        y,
        block_size,
        drop_incomplete=drop_incomplete
    ):
        if use_edge_filter:
            assert edge_mask_bool is not None

            block_edges = edge_mask_bool[y0:y1, x0:x1]
            num_edge = int(block_edges.sum())

            # 必须含有至少一个边缘像素
            if num_edge <= 0:
                continue

            edge_ratio = num_edge / float(block_edges.size)

            if edge_ratio < min_edge_ratio:
                continue

        ratio = compute_block_ratio(
            block,
            mode=mode,
            p_low=p_low,
            p_high=p_high,
            eps=eps
        )

        score = 20.0 * np.log10(ratio) if log_db else ratio
        scores.append(float(score))

    # 全图模式下，如果因为 drop_incomplete 导致没有 block，则退化为整图计算一次。
    # 边缘 block 模式下，不做 fallback；没有符合条件的边缘 block 就返回空数组。
    if len(scores) == 0 and not use_edge_filter:
        ratio = compute_block_ratio(
            y,
            mode=mode,
            p_low=p_low,
            p_high=p_high,
            eps=eps
        )
        score = 20.0 * np.log10(ratio) if log_db else ratio
        scores.append(float(score))

    return np.asarray(scores, dtype=np.float32)


def summarize_eme_scores(scores: np.ndarray, mode: str) -> dict[str, float]:
    prefix = f"eme_{mode}"

    if scores.size == 0:
        return {
            prefix: 0.0,
            f"{prefix}_block_std": 0.0,
            f"{prefix}_block_median": 0.0,
            f"{prefix}_block_p75": 0.0,
            f"{prefix}_block_p90": 0.0,
            f"{prefix}_block_p95": 0.0,
            f"{prefix}_block_max": 0.0,
            f"{prefix}_num_blocks": 0.0,
        }

    return {
        prefix: float(np.mean(scores)),
        f"{prefix}_block_std": float(np.std(scores)),
        f"{prefix}_block_median": float(np.median(scores)),
        f"{prefix}_block_p75": float(np.percentile(scores, 75)),
        f"{prefix}_block_p90": float(np.percentile(scores, 90)),
        f"{prefix}_block_p95": float(np.percentile(scores, 95)),
        f"{prefix}_block_max": float(np.max(scores)),
        f"{prefix}_num_blocks": float(scores.size),
    }


def summarize_raw_contrast_scores(scores: np.ndarray, mode: str) -> dict[str, float]:
    prefix = f"raw_contrast_{mode}"

    if scores.size == 0:
        return {
            prefix: 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_num_blocks": 0.0,
        }

    return {
        prefix: float(np.mean(scores)),
        f"{prefix}_p90": float(np.percentile(scores, 90)),
        f"{prefix}_num_blocks": float(scores.size),
    }


def summarize_edge_eme_scores(scores: np.ndarray, mode: str) -> dict[str, float]:
    prefix = f"edge_eme_{mode}"

    if scores.size == 0:
        return {
            prefix: 0.0,
            f"{prefix}_block_p90": 0.0,
            f"{prefix}_num_blocks": 0.0,
        }

    return {
        prefix: float(np.mean(scores)),
        f"{prefix}_block_p90": float(np.percentile(scores, 90)),
        f"{prefix}_num_blocks": float(scores.size),
    }


def summarize_edge_raw_contrast_scores(scores: np.ndarray, mode: str) -> dict[str, float]:
    prefix = f"edge_raw_contrast_{mode}"

    if scores.size == 0:
        return {
            prefix: 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_num_blocks": 0.0,
        }

    return {
        prefix: float(np.mean(scores)),
        f"{prefix}_p90": float(np.percentile(scores, 90)),
        f"{prefix}_num_blocks": float(scores.size),
    }


# =============================================================================
# Metric pipeline
# =============================================================================

def build_param_info(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "luma": args.luma,
        "linearize_srgb": int(args.linearize_srgb),

        "rms_window": ensure_odd_kernel(args.rms_window),
        "rms_filter": args.rms_filter,
        "rms_sigma": args.rms_sigma,
        "rms_eps": args.rms_eps,
        "rms_valid_min_mean": args.rms_valid_min_mean,

        "eme_mode": args.eme_mode,
        "block_size": get_block_size(args),
        "eme_eps": args.eme_eps,
        "eme_p_low": args.p_low,
        "eme_p_high": args.p_high,
        "eme_drop_incomplete": int(args.drop_incomplete),

        "edge_method": args.edge_method,
        "edge_canny_low": args.edge_canny_low,
        "edge_canny_high": args.edge_canny_high,
        "edge_sobel_ksize": args.edge_sobel_ksize,
        "edge_sobel_thresh": args.edge_sobel_thresh,
        "edge_min_block_ratio": args.edge_min_block_ratio,
    }


def compute_metrics_for_image(image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    img = imread_unicode(str(image_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    y = to_luminance_float01(
        img,
        luma=args.luma,
        linearize_srgb=args.linearize_srgb
    )

    h, w = y.shape
    block_size = get_block_size(args)

    metrics: dict[str, Any] = {
        "height": int(h),
        "width": int(w),
    }

    # Local RMS contrast
    rms_map, local_mean, _ = compute_local_rms_contrast(
        y,
        window=args.rms_window,
        filter_type=args.rms_filter,
        sigma=args.rms_sigma,
        eps=args.rms_eps
    )

    valid_mask = np.isfinite(rms_map)

    if args.rms_valid_min_mean > 0:
        valid_mask = valid_mask & (local_mean >= args.rms_valid_min_mean)

    metrics.update(summarize_contrast_map(rms_map, valid_mask))

    # 全图 block-wise EME / Raw Contrast
    for mode in get_eme_modes(args.eme_mode):
        eme_scores = compute_block_contrast_scores(
            y,
            block_size=block_size,
            mode=mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete,
            log_db=True,
            edge_mask=None
        )
        metrics.update(summarize_eme_scores(eme_scores, mode))

        raw_scores = compute_block_contrast_scores(
            y,
            block_size=block_size,
            mode=mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete,
            log_db=False,
            edge_mask=None
        )
        metrics.update(summarize_raw_contrast_scores(raw_scores, mode))

    # 边缘检测
    edge_mask = detect_edges(
        y,
        method=args.edge_method,
        canny_low=args.edge_canny_low,
        canny_high=args.edge_canny_high,
        sobel_ksize=args.edge_sobel_ksize,
        sobel_thresh=args.edge_sobel_thresh
    )
    edge_bool = edge_mask.astype(bool)

    # 边缘像素位置的 RMS contrast；同时沿用 rms_valid_min_mean 的 valid_mask。
    edge_rms_values = rms_map[valid_mask & edge_bool]
    metrics.update(summarize_edge_rms_contrast(edge_rms_values))

    # 含边缘像素 block 的 EME / Raw Contrast
    for mode in get_eme_modes(args.eme_mode):
        edge_eme_scores = compute_block_contrast_scores(
            y,
            block_size=block_size,
            mode=mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete,
            log_db=True,
            edge_mask=edge_mask,
            min_edge_ratio=args.edge_min_block_ratio
        )
        metrics.update(summarize_edge_eme_scores(edge_eme_scores, mode))

        edge_raw_scores = compute_block_contrast_scores(
            y,
            block_size=block_size,
            mode=mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete,
            log_db=False,
            edge_mask=edge_mask,
            min_edge_ratio=args.edge_min_block_ratio
        )
        metrics.update(summarize_edge_raw_contrast_scores(edge_raw_scores, mode))

    return metrics


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None

    if not np.isfinite(v):
        return None

    return v


def safe_delta(enh: Any, orig: Any) -> Any:
    e = finite_float(enh)
    o = finite_float(orig)

    if e is None or o is None:
        return ""

    return e - o


def safe_ratio(enh: Any, orig: Any, eps: float = 1e-12) -> Any:
    e = finite_float(enh)
    o = finite_float(orig)

    if e is None or o is None:
        return ""

    if abs(o) < eps:
        return ""

    return e / o


def build_comparison_row(
    key: str,
    original_path: Path,
    enhanced_path: Path,
    original_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    args: argparse.Namespace
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "original_image": original_path.name,
        "enhanced_image": enhanced_path.name,
        "original_path": str(original_path),
        "enhanced_path": str(enhanced_path),

        "orig_height": original_metrics.get("height", ""),
        "orig_width": original_metrics.get("width", ""),
        "enh_height": enhanced_metrics.get("height", ""),
        "enh_width": enhanced_metrics.get("width", ""),
    }

    row.update(build_param_info(args))

    metric_keys = sorted(
        list(
            (
                set(original_metrics.keys()) |
                set(enhanced_metrics.keys())
            ) - {"height", "width"}
        )
    )

    for k in metric_keys:
        row[f"orig_{k}"] = original_metrics.get(k, "")
        row[f"enh_{k}"] = enhanced_metrics.get(k, "")

    # RMS delta/gain
    rms_keys = [
        "rms_mean",
        "rms_map_std",
        "rms_median",
        "rms_p75",
        "rms_p90",
        "rms_p95",
        "rms_max",
    ]

    for k in rms_keys:
        row[f"{k}_delta"] = safe_delta(
            enhanced_metrics.get(k, ""),
            original_metrics.get(k, "")
        )
        row[f"{k}_gain"] = safe_ratio(
            enhanced_metrics.get(k, ""),
            original_metrics.get(k, "")
        )

    row["rms_valid_ratio_delta"] = safe_delta(
        enhanced_metrics.get("rms_valid_ratio", ""),
        original_metrics.get("rms_valid_ratio", "")
    )

    # Edge RMS delta/gain
    edge_rms_keys = [
        "edge_rms_mean",
        "edge_rms_median",
        "edge_rms_p90",
    ]

    for k in edge_rms_keys:
        row[f"{k}_delta"] = safe_delta(
            enhanced_metrics.get(k, ""),
            original_metrics.get(k, "")
        )
        row[f"{k}_gain"] = safe_ratio(
            enhanced_metrics.get(k, ""),
            original_metrics.get(k, "")
        )

    # EME delta
    for mode in get_eme_modes(args.eme_mode):
        eme_keys = [
            f"eme_{mode}",
            f"eme_{mode}_block_std",
            f"eme_{mode}_block_median",
            f"eme_{mode}_block_p75",
            f"eme_{mode}_block_p90",
            f"eme_{mode}_block_p95",
            f"eme_{mode}_block_max",
        ]

        for k in eme_keys:
            row[f"{k}_delta_db"] = safe_delta(
                enhanced_metrics.get(k, ""),
                original_metrics.get(k, "")
            )

        edge_eme_keys = [
            f"edge_eme_{mode}",
            f"edge_eme_{mode}_block_p90",
        ]

        for k in edge_eme_keys:
            row[f"{k}_delta_db"] = safe_delta(
                enhanced_metrics.get(k, ""),
                original_metrics.get(k, "")
            )

    return row


# =============================================================================
# CSV field order
# =============================================================================

def get_single_csv_fields() -> list[str]:
    fields = [
        "key",
        "image",
        "path",
        "height",
        "width",

        "luma",
        "linearize_srgb",

        "rms_window",
        "rms_filter",
        "rms_sigma",
        "rms_eps",
        "rms_valid_min_mean",

        "rms_valid_ratio",
        "rms_mean",
        "rms_map_std",
        "rms_median",
        "rms_p75",
        "rms_p90",
        "rms_p95",
        "rms_max",

        "eme_mode",
        "block_size",
        "eme_eps",
        "eme_p_low",
        "eme_p_high",
        "eme_drop_incomplete",

        "edge_method",
        "edge_canny_low",
        "edge_canny_high",
        "edge_sobel_ksize",
        "edge_sobel_thresh",
        "edge_min_block_ratio",

        "edge_rms_mean",
        "edge_rms_median",
        "edge_rms_p90",
        "edge_rms_num_pixels",
    ]

    for mode in ["classic", "percentile"]:
        fields.extend([
            f"eme_{mode}",
            f"eme_{mode}_block_std",
            f"eme_{mode}_block_median",
            f"eme_{mode}_block_p75",
            f"eme_{mode}_block_p90",
            f"eme_{mode}_block_p95",
            f"eme_{mode}_block_max",
            f"eme_{mode}_num_blocks",
        ])

        fields.extend([
            f"raw_contrast_{mode}",
            f"raw_contrast_{mode}_p90",
            f"raw_contrast_{mode}_num_blocks",
        ])

        fields.extend([
            f"edge_eme_{mode}",
            f"edge_eme_{mode}_block_p90",
            f"edge_eme_{mode}_num_blocks",
        ])

        fields.extend([
            f"edge_raw_contrast_{mode}",
            f"edge_raw_contrast_{mode}_p90",
            f"edge_raw_contrast_{mode}_num_blocks",
        ])

    return fields


def get_comparison_csv_fields() -> list[str]:
    fields = [
        "key",
        "original_image",
        "enhanced_image",
        "original_path",
        "enhanced_path",

        "orig_height",
        "orig_width",
        "enh_height",
        "enh_width",

        "luma",
        "linearize_srgb",

        "rms_window",
        "rms_filter",
        "rms_sigma",
        "rms_eps",
        "rms_valid_min_mean",

        "eme_mode",
        "block_size",
        "eme_eps",
        "eme_p_low",
        "eme_p_high",
        "eme_drop_incomplete",

        "edge_method",
        "edge_canny_low",
        "edge_canny_high",
        "edge_sobel_ksize",
        "edge_sobel_thresh",
        "edge_min_block_ratio",

        "orig_rms_valid_ratio",
        "enh_rms_valid_ratio",
        "rms_valid_ratio_delta",

        "orig_rms_mean",
        "enh_rms_mean",
        "rms_mean_delta",
        "rms_mean_gain",

        "orig_rms_median",
        "enh_rms_median",
        "rms_median_delta",
        "rms_median_gain",

        "orig_rms_p75",
        "enh_rms_p75",
        "rms_p75_delta",
        "rms_p75_gain",

        "orig_rms_p90",
        "enh_rms_p90",
        "rms_p90_delta",
        "rms_p90_gain",

        "orig_rms_p95",
        "enh_rms_p95",
        "rms_p95_delta",
        "rms_p95_gain",

        "orig_edge_rms_mean",
        "enh_edge_rms_mean",
        "edge_rms_mean_delta",
        "edge_rms_mean_gain",

        "orig_edge_rms_median",
        "enh_edge_rms_median",
        "edge_rms_median_delta",
        "edge_rms_median_gain",

        "orig_edge_rms_p90",
        "enh_edge_rms_p90",
        "edge_rms_p90_delta",
        "edge_rms_p90_gain",

        "orig_edge_rms_num_pixels",
        "enh_edge_rms_num_pixels",
    ]

    for mode in ["classic", "percentile"]:
        fields.extend([
            f"orig_eme_{mode}",
            f"enh_eme_{mode}",
            f"eme_{mode}_delta_db",

            f"orig_eme_{mode}_block_median",
            f"enh_eme_{mode}_block_median",
            f"eme_{mode}_block_median_delta_db",

            f"orig_eme_{mode}_block_p90",
            f"enh_eme_{mode}_block_p90",
            f"eme_{mode}_block_p90_delta_db",

            f"orig_eme_{mode}_block_p95",
            f"enh_eme_{mode}_block_p95",
            f"eme_{mode}_block_p95_delta_db",
        ])

        fields.extend([
            f"orig_raw_contrast_{mode}",
            f"enh_raw_contrast_{mode}",
            f"orig_raw_contrast_{mode}_p90",
            f"enh_raw_contrast_{mode}_p90",
            f"orig_raw_contrast_{mode}_num_blocks",
            f"enh_raw_contrast_{mode}_num_blocks",
        ])

        fields.extend([
            f"orig_edge_eme_{mode}",
            f"enh_edge_eme_{mode}",
            f"edge_eme_{mode}_delta_db",

            f"orig_edge_eme_{mode}_block_p90",
            f"enh_edge_eme_{mode}_block_p90",
            f"edge_eme_{mode}_block_p90_delta_db",

            f"orig_edge_eme_{mode}_num_blocks",
            f"enh_edge_eme_{mode}_num_blocks",
        ])

        fields.extend([
            f"orig_edge_raw_contrast_{mode}",
            f"enh_edge_raw_contrast_{mode}",
            f"orig_edge_raw_contrast_{mode}_p90",
            f"enh_edge_raw_contrast_{mode}_p90",
            f"orig_edge_raw_contrast_{mode}_num_blocks",
            f"enh_edge_raw_contrast_{mode}_num_blocks",
        ])

    return fields


# =============================================================================
# Pair metric dashboard
# =============================================================================

def get_pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    cjk_candidates = [
        "Noto Sans CJK SC", "Noto Sans SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Source Han Sans SC", "SimHei", "Microsoft YaHei",
        "AR PL UMing CN", "Droid Sans Fallback",
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for name in cjk_candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break

    plt.rcParams["axes.unicode_minus"] = False

    return plt


def sanitize_key_to_stem(key: str) -> str:
    stem = Path(key).with_suffix("").as_posix()
    stem = stem.replace("/", "__").replace("\\", "__")
    stem = stem.replace(":", "_")

    return stem


def image_to_rgb_float_for_plot(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        y = normalize_to_float01(img)
        return np.stack([y, y, y], axis=-1).astype(np.float32)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]

        f = normalize_to_float01(img)

        # BGR -> RGB
        return np.clip(f[:, :, ::-1], 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported image shape: {img.shape}")

def save_edge_visualizations_for_image(
    image_path: Path,
    args: argparse.Namespace,
    edge_mask_path: Path | None = None,
    overlay_path: Path | None = None,
    overlay_alpha: float = 0.85
) -> bool:
    """
    保存边缘检测结果。

    edge_mask_path:
        保存黑底白边的二值边缘图。

    overlay_path:
        保存红色边缘叠加图，方便检查边缘是否符合预期。
    """
    try:
        if edge_mask_path is None and overlay_path is None:
            return True

        img = imread_unicode(str(image_path))

        if img is None:
            return False

        y = to_luminance_float01(
            img,
            luma=args.luma,
            linearize_srgb=args.linearize_srgb
        )

        edge_mask = detect_edges(
            y,
            method=args.edge_method,
            canny_low=args.edge_canny_low,
            canny_high=args.edge_canny_high,
            sobel_ksize=args.edge_sobel_ksize,
            sobel_thresh=args.edge_sobel_thresh
        )

        ok_all = True

        # 1. 保存二值边缘图：黑底白边
        if edge_mask_path is not None:
            edge_u8 = (edge_mask.astype(np.uint8) * 255)
            ok = imwrite_unicode(str(edge_mask_path), edge_u8)
            ok_all = ok_all and ok

        # 2. 保存红色边缘叠加图
        if overlay_path is not None:
            overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))

            rgb = image_to_rgb_float_for_plot(img)
            rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

            out_rgb = rgb_u8.copy()
            mask_bool = edge_mask.astype(bool)

            edge_color_rgb = np.array([255, 0, 0], dtype=np.float32)

            out_rgb[mask_bool] = (
                (1.0 - overlay_alpha) * out_rgb[mask_bool].astype(np.float32)
                + overlay_alpha * edge_color_rgb
            ).astype(np.uint8)

            # OpenCV 写图用 BGR
            out_bgr = out_rgb[:, :, ::-1]

            ok = imwrite_unicode(str(overlay_path), out_bgr)
            ok_all = ok_all and ok

        return ok_all

    except Exception as e:
        print(f"  [Warning] Failed to save edge visualization: {image_path}, error={e}")
        return False

def resize_rgb_for_plot(rgb: np.ndarray, max_side: int = 1200) -> np.ndarray:
    if max_side <= 0:
        return rgb

    h, w = rgb.shape[:2]
    m = max(h, w)

    if m <= max_side:
        return rgb

    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def metric_value(d: dict[str, Any], key: str) -> float:
    v = finite_float(d.get(key, ""))

    return float("nan") if v is None else float(v)


def fmt_metric(v: Any, nd: int = 3, suffix: str = "") -> str:
    fv = finite_float(v)

    if fv is None:
        return "N/A"

    return f"{fv:.{nd}f}{suffix}"


def set_bar_ylim(ax, values: list[float], ref_line: float | None = None) -> None:
    finite_vals = [float(v) for v in values if np.isfinite(v)]

    if ref_line is not None:
        finite_vals.append(float(ref_line))

    finite_vals.append(0.0)

    lo = min(finite_vals)
    hi = max(finite_vals)

    if abs(hi - lo) < 1e-12:
        if hi == 0:
            lo, hi = -1.0, 1.0
        else:
            margin = abs(hi) * 0.2
            lo -= margin
            hi += margin
    else:
        margin = (hi - lo) * 0.18
        lo -= margin
        hi += margin

    lo = min(lo, 0.0)
    ax.set_ylim(lo, hi)


def add_bar_labels(ax, bars, values: list[float], fmt: str) -> None:
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * 0.015

    for bar, v in zip(bars, values):
        if not np.isfinite(v):
            continue

        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0

        if h >= 0:
            y = h + offset
            va = "bottom"
        else:
            y = h - offset
            va = "top"

        ax.text(
            x,
            y,
            fmt.format(v),
            ha="center",
            va=va,
            fontsize=7
        )


def plot_grouped_bars(
    ax,
    names: list[str],
    orig_vals: list[float],
    enh_vals: list[float],
    title: str,
    ylabel: str,
    value_fmt: str
) -> None:
    x = np.arange(len(names))
    width = 0.36

    bars1 = ax.bar(
        x - width / 2,
        orig_vals,
        width,
        label="Original",
        color="#4C78A8"
    )
    bars2 = ax.bar(
        x + width / 2,
        enh_vals,
        width,
        label="Enhanced",
        color="#F58518"
    )

    set_bar_ylim(ax, orig_vals + enh_vals)

    add_bar_labels(ax, bars1, orig_vals, value_fmt)
    add_bar_labels(ax, bars2, enh_vals, value_fmt)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)


def plot_single_bars(
    ax,
    names: list[str],
    vals: list[float],
    title: str,
    ylabel: str,
    value_fmt: str,
    ref_line: float | None = None,
    color_mode: str = "normal"
) -> None:
    x = np.arange(len(names))

    colors = []

    for v in vals:
        if not np.isfinite(v):
            colors.append("#999999")
        elif color_mode == "gain":
            colors.append("#D62728" if v >= 1.0 else "#1F77B4")
        elif color_mode == "delta":
            colors.append("#D62728" if v >= 0.0 else "#1F77B4")
        else:
            colors.append("#4C78A8")

    bars = ax.bar(x, vals, width=0.42, color=colors)

    if len(x) > 0:
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    set_bar_ylim(ax, vals, ref_line=ref_line)

    if ref_line is not None:
        ax.axhline(
            ref_line,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75
        )

    add_bar_labels(ax, bars, vals, value_fmt)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def save_pair_metric_dashboard(
    key: str,
    original_path: Path,
    enhanced_path: Path,
    original_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    comparison_row: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace
) -> bool:
    """
    生成单对指标仪表盘：
        上行：原图/增强图
        下行：RMS、RMS Gain、EME、EME Delta、Raw Contrast、逐块 EME Delta
    """
    try:
        plt = get_pyplot()
        dashboard_eme_mode = get_dashboard_eme_mode(args)
        block_size = get_block_size(args)

        img_orig = imread_unicode(str(original_path))
        img_enh = imread_unicode(str(enhanced_path))

        if img_orig is None or img_enh is None:
            return False

        rgb_orig = resize_rgb_for_plot(
            image_to_rgb_float_for_plot(img_orig),
            max_side=args.pair_figure_max_image_side
        )
        rgb_enh = resize_rgb_for_plot(
            image_to_rgb_float_for_plot(img_enh),
            max_side=args.pair_figure_max_image_side
        )

        fig = plt.figure(figsize=(22, 9.8), dpi=args.pair_figure_dpi)

        gs_top = fig.add_gridspec(
            1,
            2,
            left=0.05,
            right=0.95,
            top=0.92,
            bottom=0.55,
            wspace=0.02
        )

        gs_bot = fig.add_gridspec(
            1,
            6,
            left=0.05,
            right=0.95,
            top=0.50,
            bottom=0.15,
            wspace=0.48,
            width_ratios=[1.15, 1.15, 1.15, 1.15, 1.15, 3.0]
        )

        # 第一行：图片
        ax_img_orig = fig.add_subplot(gs_top[0, 0])
        ax_img_enh = fig.add_subplot(gs_top[0, 1])

        ax_img_orig.imshow(rgb_orig)
        ax_img_orig.set_title(f"Original: {original_path.name}", fontsize=11)
        ax_img_orig.axis("off")

        ax_img_enh.imshow(rgb_enh)
        ax_img_enh.set_title(f"Enhanced: {enhanced_path.name}", fontsize=11)
        ax_img_enh.axis("off")

        # RMS 原/增强：mean, p90, edge mean
        ax_rms = fig.add_subplot(gs_bot[0, 0])

        rms_names = ["mean", "p90", "edge\nmean"]
        rms_keys = [
            "rms_mean",
            "rms_p90",
            "edge_rms_mean",
        ]

        plot_grouped_bars(
            ax=ax_rms,
            names=rms_names,
            orig_vals=[metric_value(original_metrics, k) for k in rms_keys],
            enh_vals=[metric_value(enhanced_metrics, k) for k in rms_keys],
            title="Local RMS Contrast",
            ylabel="RMS contrast",
            value_fmt="{:.4f}"
        )

        # RMS Gain：mean, p90, edge mean
        ax_gain = fig.add_subplot(gs_bot[0, 1])

        gain_names = ["mean", "p90", "edge\nmean"]
        gain_keys = [
            "rms_mean_gain",
            "rms_p90_gain",
            "edge_rms_mean_gain",
        ]

        plot_single_bars(
            ax=ax_gain,
            names=gain_names,
            vals=[metric_value(comparison_row, k) for k in gain_keys],
            title="Local RMS Gain",
            ylabel="enhanced / original",
            value_fmt="{:.3f}x",
            ref_line=1.0,
            color_mode="gain"
        )

        # EME 原/增强：mean, p90, edge mean
        ax_eme = fig.add_subplot(gs_bot[0, 2])

        eme_names = ["mean", "p90", "edge\nmean"]
        eme_keys = [
            f"eme_{dashboard_eme_mode}",
            f"eme_{dashboard_eme_mode}_block_p90",
            f"edge_eme_{dashboard_eme_mode}",
        ]

        plot_grouped_bars(
            ax=ax_eme,
            names=eme_names,
            orig_vals=[metric_value(original_metrics, k) for k in eme_keys],
            enh_vals=[metric_value(enhanced_metrics, k) for k in eme_keys],
            title=f"EME ({dashboard_eme_mode})",
            ylabel="EME / dB",
            value_fmt="{:.3f}"
        )

        # EME Delta：mean, p90, edge mean
        ax_eme_delta = fig.add_subplot(gs_bot[0, 3])

        eme_delta_names = ["mean", "p90", "edge\nmean"]
        eme_delta_keys = [
            f"eme_{dashboard_eme_mode}_delta_db",
            f"eme_{dashboard_eme_mode}_block_p90_delta_db",
            f"edge_eme_{dashboard_eme_mode}_delta_db",
        ]

        plot_single_bars(
            ax=ax_eme_delta,
            names=eme_delta_names,
            vals=[metric_value(comparison_row, k) for k in eme_delta_keys],
            title=f"EME Delta ({dashboard_eme_mode})",
            ylabel="enhanced - original / dB",
            value_fmt="{:+.3f}",
            ref_line=0.0,
            color_mode="delta"
        )

        # Raw Contrast：mean, p90, edge mean
        ax_raw = fig.add_subplot(gs_bot[0, 4])

        raw_names = ["mean", "p90", "edge\nmean"]
        raw_keys = [
            f"raw_contrast_{dashboard_eme_mode}",
            f"raw_contrast_{dashboard_eme_mode}_p90",
            f"edge_raw_contrast_{dashboard_eme_mode}",
        ]

        plot_grouped_bars(
            ax=ax_raw,
            names=raw_names,
            orig_vals=[metric_value(original_metrics, k) for k in raw_keys],
            enh_vals=[metric_value(enhanced_metrics, k) for k in raw_keys],
            title=f"Raw Contrast ({dashboard_eme_mode})",
            ylabel="ratio, no log",
            value_fmt="{:.2f}"
        )

        # 逐块 EME Delta
        ax_eme_line = fig.add_subplot(gs_bot[0, 5])

        y_orig = to_luminance_float01(
            img_orig,
            luma=args.luma,
            linearize_srgb=args.linearize_srgb
        )
        y_enh = to_luminance_float01(
            img_enh,
            luma=args.luma,
            linearize_srgb=args.linearize_srgb
        )

        scores_orig = compute_block_contrast_scores(
            y_orig,
            block_size=block_size,
            mode=dashboard_eme_mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete,
            log_db=True,
            edge_mask=None
        )
        scores_enh = compute_block_contrast_scores(
            y_enh,
            block_size=block_size,
            mode=dashboard_eme_mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete,
            log_db=True,
            edge_mask=None
        )

        n_blocks = min(len(scores_orig), len(scores_enh))
        scores_delta = scores_enh[:n_blocks] - scores_orig[:n_blocks]
        block_indices = np.arange(n_blocks)

        pos_mask = scores_delta >= 0

        ax_eme_line.bar(
            block_indices[pos_mask],
            scores_delta[pos_mask],
            color="#F58518",
            alpha=0.6,
            width=1.0,
            linewidth=0
        )
        ax_eme_line.bar(
            block_indices[~pos_mask],
            scores_delta[~pos_mask],
            color="#4C78A8",
            alpha=0.6,
            width=1.0,
            linewidth=0
        )
        ax_eme_line.axhline(
            0,
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.5
        )

        ax_eme_line.set_title(
            f"EME Delta per Block ({dashboard_eme_mode})",
            fontsize=10
        )
        ax_eme_line.set_xlabel("Block index", fontsize=8)
        ax_eme_line.set_ylabel("增强 - 原图 / dB", fontsize=9)
        ax_eme_line.grid(axis="y", linestyle="--", alpha=0.35)

        eme_summary = (
            f"EME {dashboard_eme_mode} delta="
            f"{fmt_metric(comparison_row.get(f'eme_{dashboard_eme_mode}_delta_db'), 3, ' dB')}"
        )
        edge_eme_summary = (
            f"edge-block EME delta="
            f"{fmt_metric(comparison_row.get(f'edge_eme_{dashboard_eme_mode}_delta_db'), 3, ' dB')}"
        )

        fig.suptitle(
            (
                f"{key} | "
                f"RMS mean gain={fmt_metric(comparison_row.get('rms_mean_gain'), 3, 'x')}, "
                f"RMS edge mean gain={fmt_metric(comparison_row.get('edge_rms_mean_gain'), 3, 'x')}, "
                f"{eme_summary}, "
                f"{edge_eme_summary}"
            ),
            fontsize=12
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)

        return True

    except Exception as e:
        print(f"  [Warning] Failed to save pair metric figure: {output_path}, error={e}")
        return False


# =============================================================================
# Summary
# =============================================================================

def collect_numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals: list[float] = []

    for row in rows:
        v = finite_float(row.get(key, ""))
        if v is not None:
            vals.append(v)

    return np.asarray(vals, dtype=np.float64)


def print_summary(comparison_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("-" * 80)
    print("Dataset Comparison Summary")
    print("-" * 80)

    for key in [
        "rms_mean_gain",
        "rms_p90_gain",
        "edge_rms_mean_gain",
    ]:
        vals = collect_numeric(comparison_rows, key)

        if vals.size > 0:
            print(
                f"{key:36s}: "
                f"mean={np.mean(vals):.4f}, "
                f"median={np.median(vals):.4f}"
            )

    vals = collect_numeric(comparison_rows, "edge_rms_mean_delta")

    if vals.size > 0:
        print(
            f"{'edge_rms_mean_delta':36s}: "
            f"mean={np.mean(vals):.4f}, "
            f"median={np.median(vals):.4f}"
        )

    for mode in get_eme_modes(args.eme_mode):
        for key in [
            f"eme_{mode}_delta_db",
            f"edge_eme_{mode}_delta_db",
            f"edge_eme_{mode}_block_p90_delta_db",
        ]:
            vals = collect_numeric(comparison_rows, key)

            if vals.size > 0:
                print(
                    f"{key:36s}: "
                    f"mean={np.mean(vals):.4f} dB, "
                    f"median={np.median(vals):.4f} dB"
                )

        for prefix in ["orig", "enh"]:
            for base in [
                f"raw_contrast_{mode}",
                f"edge_raw_contrast_{mode}",
            ]:
                for suffix in ["", "_p90"]:
                    k = f"{prefix}_{base}{suffix}"
                    vals = collect_numeric(comparison_rows, k)

                    if vals.size > 0:
                        tag = f"{base}{suffix} ({prefix})"
                        print(
                            f"{tag:36s}: "
                            f"mean={np.mean(vals):.4f}, "
                            f"median={np.median(vals):.4f}"
                        )

    print("-" * 80)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute local RMS contrast, EME and raw contrast for original/enhanced image folders."
    )

    parser.add_argument(
        "--original",
        "--input-original",
        dest="input_original",
        type=str,
        required=True,
        help="Original image folder."
    )
    parser.add_argument(
        "--enhanced",
        "--input-enhanced",
        dest="input_enhanced",
        type=str,
        required=True,
        help="Enhanced image folder."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./test/rms_eme_eval",
        help="Output folder."
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively collect images."
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="basename",
        choices=["basename", "relative"],
        help="Image matching mode."
    )

    # Luminance
    parser.add_argument(
        "--luma",
        type=str,
        default="bt709",
        choices=["bt601", "bt709"],
        help="Luminance conversion."
    )
    parser.add_argument(
        "--linearize-srgb",
        action="store_true",
        help="Apply inverse sRGB transfer before luminance conversion."
    )

    # RMS
    parser.add_argument(
        "--rms-window",
        type=int,
        default=15,
        help="Local RMS window size."
    )
    parser.add_argument(
        "--rms-filter",
        type=str,
        default="gaussian",
        choices=["gaussian", "box"],
        help="Local RMS filter type."
    )
    parser.add_argument(
        "--rms-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for RMS. 0 lets OpenCV choose automatically."
    )
    parser.add_argument(
        "--rms-eps",
        type=float,
        default=1e-4,
        help="Epsilon in local RMS denominator."
    )
    parser.add_argument(
        "--rms-valid-min-mean",
        type=float,
        default=0.0,
        help="Exclude pixels whose local mean is below this threshold. Also applied to edge RMS."
    )

    # EME / Raw contrast
    parser.add_argument(
        "--eme-mode",
        type=str,
        default="classic",
        choices=["classic", "percentile", "both"],
        help="EME/raw contrast mode."
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Square block size for EME/raw contrast."
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=5.0,
        help="Low percentile for percentile mode."
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=95.0,
        help="High percentile for percentile mode."
    )
    parser.add_argument(
        "--eme-eps",
        type=float,
        default=1.0 / 255.0,
        help="Epsilon for EME/raw ratio."
    )
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Drop incomplete border blocks for EME/raw contrast."
    )

    # Edge detection
    parser.add_argument(
        "--edge-method",
        type=str,
        default="canny",
        choices=["canny", "sobel"],
        help="Edge detection method for edge-based metrics."
    )
    parser.add_argument(
        "--edge-canny-low",
        type=float,
        default=0.15,
        help="Canny low threshold, relative to [0,1]."
    )
    parser.add_argument(
        "--edge-canny-high",
        type=float,
        default=0.40,
        help="Canny high threshold, relative to [0,1]."
    )
    parser.add_argument(
        "--edge-sobel-ksize",
        type=int,
        default=3,
        help="Sobel kernel size."
    )
    parser.add_argument(
        "--edge-sobel-thresh",
        type=float,
        default=0.1,
        help="Sobel gradient magnitude threshold."
    )
    parser.add_argument(
        "--edge-min-block-ratio",
        type=float,
        default=0.01,
        help=(
            "Minimum edge pixel ratio in a block to compute edge-block EME/raw contrast. "
            "0.0 means any block containing at least one edge pixel."
        )
    )
    # Edge map output
    parser.add_argument(
        "--save-edge-maps",
        action="store_true",
        help="Save binary edge maps for original/enhanced images."
    )
    parser.add_argument(
        "--save-edge-overlays",
        action="store_true",
        help="Save red edge overlay images for original/enhanced images."
    )
    parser.add_argument(
        "--edge-map-dir",
        type=str,
        default=None,
        help="Edge map output folder. Default: output/edge_maps."
    )
    parser.add_argument(
        "--edge-overlay-alpha",
        type=float,
        default=0.85,
        help="Alpha of red edge overlay, in [0,1]."
    )
    # Pair figure
    parser.add_argument(
        "--save-pair-metric-figures",
        action="store_true",
        help="Save one combined metric figure for each original/enhanced pair."
    )
    parser.add_argument(
        "--pair-figure-dir",
        type=str,
        default=None,
        help="Pair metric figure output folder. Default: output/pair_metric_figures."
    )
    parser.add_argument(
        "--pair-figure-dpi",
        type=int,
        default=160,
        help="DPI for pair metric figures."
    )
    parser.add_argument(
        "--pair-figure-max-image-side",
        type=int,
        default=1200,
        help="Max image side length used for display in pair figures."
    )

    args = parser.parse_args()

    if args.p_high <= args.p_low:
        print("[Error] --p-high must be larger than --p-low")
        sys.exit(1)

    if args.block_size <= 0:
        print("[Error] --block-size must be > 0")
        sys.exit(1)

    if not (0.0 <= args.edge_min_block_ratio <= 1.0):
        print("[Error] --edge-min-block-ratio must be in [0, 1]")
        sys.exit(1)

    if not (0.0 <= args.edge_overlay_alpha <= 1.0):
        print("[Error] --edge-overlay-alpha must be in [0, 1]")
        sys.exit(1)

    if args.edge_method == "canny" and args.edge_canny_high <= args.edge_canny_low:
        print("[Error] --edge-canny-high must be larger than --edge-canny-low")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_dict, original_duplicates = build_image_dict(
        args.input_original,
        recursive=args.recursive,
        match_mode=args.match_mode
    )

    enhanced_dict, enhanced_duplicates = build_image_dict(
        args.input_enhanced,
        recursive=args.recursive,
        match_mode=args.match_mode
    )

    if original_duplicates:
        print(f"[Warning] Duplicate keys in original folder: {len(original_duplicates)}")
        write_text_list(
            output_dir / "duplicate_keys_original.txt",
            sorted(original_duplicates)
        )

    if enhanced_duplicates:
        print(f"[Warning] Duplicate keys in enhanced folder: {len(enhanced_duplicates)}")
        write_text_list(
            output_dir / "duplicate_keys_enhanced.txt",
            sorted(enhanced_duplicates)
        )

    original_keys = set(original_dict.keys())
    enhanced_keys = set(enhanced_dict.keys())

    common_keys = sorted(list(original_keys & enhanced_keys))
    missing_in_original = sorted(list(enhanced_keys - original_keys))
    missing_in_enhanced = sorted(list(original_keys - enhanced_keys))

    write_text_list(output_dir / "unmatched_in_original.txt", missing_in_original)
    write_text_list(output_dir / "unmatched_in_enhanced.txt", missing_in_enhanced)

    if not common_keys:
        print("[Error] No matched images found.")
        print(f"Original folder: {args.input_original}")
        print(f"Enhanced folder: {args.input_enhanced}")
        print(f"Match mode     : {args.match_mode}")
        sys.exit(1)

    param_info = build_param_info(args)

    print("=" * 80)
    print("Local RMS Contrast + EME + Raw Contrast Pair Evaluation")
    print("=" * 80)
    print(f"Original folder      : {args.input_original}")
    print(f"Enhanced folder      : {args.input_enhanced}")
    print(f"Output               : {args.output}")
    print(f"Match mode           : {args.match_mode}")
    print(f"Original images      : {len(original_dict)}")
    print(f"Enhanced images      : {len(enhanced_dict)}")
    print(f"Matched pairs        : {len(common_keys)}")
    print(f"Missing in original  : {len(missing_in_original)}")
    print(f"Missing in enhanced  : {len(missing_in_enhanced)}")
    print("-" * 80)
    print(f"Luma                 : {args.luma}")
    print(f"Linearize sRGB       : {args.linearize_srgb}")
    print(f"RMS window/filter    : {ensure_odd_kernel(args.rms_window)} / {args.rms_filter}")
    print(f"RMS eps              : {args.rms_eps}")
    print(f"RMS valid min mean   : {args.rms_valid_min_mean}")
    print(f"EME/raw mode         : {args.eme_mode}")
    print(f"Block size           : {param_info['block_size']} x {param_info['block_size']}")
    print(f"EME/raw eps          : {args.eme_eps}")

    if args.eme_mode in ["percentile", "both"]:
        print(f"Percentile mode      : P{args.p_low} / P{args.p_high}")

    print(f"Edge method          : {args.edge_method}")
    print(f"Edge canny low/high  : {args.edge_canny_low} / {args.edge_canny_high}")
    print(f"Edge min block ratio : {args.edge_min_block_ratio}")
    print(f"Save edge maps       : {args.save_edge_maps}")
    print(f"Save edge overlays   : {args.save_edge_overlays}")
    print(f"Pair metric figures  : {args.save_pair_metric_figures}")
    print("=" * 80)

    original_rows: list[dict[str, Any]] = []
    enhanced_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for idx, key in enumerate(common_keys, 1):
        original_path = original_dict[key]
        enhanced_path = enhanced_dict[key]

        print(f"[{idx}/{len(common_keys)}] {key}")

        try:
            original_metrics = compute_metrics_for_image(original_path, args)
        except Exception as e:
            print(f"  [Warning] Failed original: {original_path}, error={e}")
            continue

        try:
            enhanced_metrics = compute_metrics_for_image(enhanced_path, args)
        except Exception as e:
            print(f"  [Warning] Failed enhanced: {enhanced_path}, error={e}")
            continue

        original_row: dict[str, Any] = {
            "key": key,
            "image": original_path.name,
            "path": str(original_path),
        }
        original_row.update(param_info)
        original_row.update(original_metrics)

        enhanced_row: dict[str, Any] = {
            "key": key,
            "image": enhanced_path.name,
            "path": str(enhanced_path),
        }
        enhanced_row.update(param_info)
        enhanced_row.update(enhanced_metrics)

        comparison_row = build_comparison_row(
            key=key,
            original_path=original_path,
            enhanced_path=enhanced_path,
            original_metrics=original_metrics,
            enhanced_metrics=enhanced_metrics,
            args=args
        )

        original_rows.append(original_row)
        enhanced_rows.append(enhanced_row)
        comparison_rows.append(comparison_row)

        # 保存边缘检测图
        if args.save_edge_maps or args.save_edge_overlays:
            edge_map_dir = (
                Path(args.edge_map_dir)
                if args.edge_map_dir
                else output_dir / "edge_maps"
            )

            safe_stem = sanitize_key_to_stem(key)

            orig_edge_mask_path = (
                edge_map_dir / "original_mask" / f"{safe_stem}_edge_mask.png"
                if args.save_edge_maps
                else None
            )
            enh_edge_mask_path = (
                edge_map_dir / "enhanced_mask" / f"{safe_stem}_edge_mask.png"
                if args.save_edge_maps
                else None
            )

            orig_edge_overlay_path = (
                edge_map_dir / "original_overlay" / f"{safe_stem}_edge_overlay.png"
                if args.save_edge_overlays
                else None
            )
            enh_edge_overlay_path = (
                edge_map_dir / "enhanced_overlay" / f"{safe_stem}_edge_overlay.png"
                if args.save_edge_overlays
                else None
            )

            ok_orig_edge = save_edge_visualizations_for_image(
                image_path=original_path,
                args=args,
                edge_mask_path=orig_edge_mask_path,
                overlay_path=orig_edge_overlay_path,
                overlay_alpha=args.edge_overlay_alpha
            )

            ok_enh_edge = save_edge_visualizations_for_image(
                image_path=enhanced_path,
                args=args,
                edge_mask_path=enh_edge_mask_path,
                overlay_path=enh_edge_overlay_path,
                overlay_alpha=args.edge_overlay_alpha
            )

            if ok_orig_edge and ok_enh_edge:
                print(f"  Saved edge visualizations: {edge_map_dir}")

        if args.save_pair_metric_figures:
            pair_figure_dir = (
                Path(args.pair_figure_dir)
                if args.pair_figure_dir
                else output_dir / "pair_metric_figures"
            )

            safe_stem = sanitize_key_to_stem(key)
            figure_path = pair_figure_dir / f"{safe_stem}_metric_dashboard.png"

            ok = save_pair_metric_dashboard(
                key=key,
                original_path=original_path,
                enhanced_path=enhanced_path,
                original_metrics=original_metrics,
                enhanced_metrics=enhanced_metrics,
                comparison_row=comparison_row,
                output_path=figure_path,
                args=args
            )

            if ok:
                print(f"  Saved pair metric figure: {figure_path}")

        msg = (
            f"  RMS mean: "
            f"{original_metrics.get('rms_mean', 0):.6f} -> "
            f"{enhanced_metrics.get('rms_mean', 0):.6f}"
        )

        rms_gain = safe_ratio(
            enhanced_metrics.get("rms_mean", ""),
            original_metrics.get("rms_mean", "")
        )

        if rms_gain != "":
            msg += f", gain={rms_gain:.4f}x"

        edge_rms_gain = safe_ratio(
            enhanced_metrics.get("edge_rms_mean", ""),
            original_metrics.get("edge_rms_mean", "")
        )

        if edge_rms_gain != "":
            msg += f", edge_rms_mean_gain={edge_rms_gain:.4f}x"

        for mode in get_eme_modes(args.eme_mode):
            k = f"eme_{mode}"

            if k in original_metrics and k in enhanced_metrics:
                delta = safe_delta(
                    enhanced_metrics.get(k, ""),
                    original_metrics.get(k, "")
                )

                if delta != "":
                    msg += (
                        f", {k}: "
                        f"{original_metrics[k]:.4f} -> "
                        f"{enhanced_metrics[k]:.4f} dB, "
                        f"delta={delta:.4f} dB"
                    )

            edge_k = f"edge_eme_{mode}"

            if edge_k in original_metrics and edge_k in enhanced_metrics:
                edge_delta = safe_delta(
                    enhanced_metrics.get(edge_k, ""),
                    original_metrics.get(edge_k, "")
                )

                if edge_delta != "":
                    msg += (
                        f", {edge_k}: "
                        f"{original_metrics[edge_k]:.4f} -> "
                        f"{enhanced_metrics[edge_k]:.4f} dB, "
                        f"delta={edge_delta:.4f} dB"
                    )

            raw_k = f"raw_contrast_{mode}"
            edge_raw_k = f"edge_raw_contrast_{mode}"

            if raw_k in original_metrics and raw_k in enhanced_metrics:
                msg += (
                    f", {raw_k}: "
                    f"{original_metrics[raw_k]:.4f} -> "
                    f"{enhanced_metrics[raw_k]:.4f}"
                )

            if edge_raw_k in original_metrics and edge_raw_k in enhanced_metrics:
                msg += (
                    f", {edge_raw_k}: "
                    f"{original_metrics[edge_raw_k]:.4f} -> "
                    f"{enhanced_metrics[edge_raw_k]:.4f}"
                )
        print(msg)

    original_csv = output_dir / "original_metrics.csv"
    enhanced_csv = output_dir / "enhanced_metrics.csv"
    comparison_csv = output_dir / "paired_comparison_metrics.csv"

    write_csv(
        original_csv,
        original_rows,
        preferred_fields=get_single_csv_fields()
    )

    write_csv(
        enhanced_csv,
        enhanced_rows,
        preferred_fields=get_single_csv_fields()
    )

    write_csv(
        comparison_csv,
        comparison_rows,
        preferred_fields=get_comparison_csv_fields()
    )

    print_summary(comparison_rows, args)

    print("Saved:")
    print(f"  Original metrics        : {original_csv}")
    print(f"  Enhanced metrics        : {enhanced_csv}")
    print(f"  Paired comparison       : {comparison_csv}")
    print(f"  Missing in enhanced     : {output_dir / 'unmatched_in_enhanced.txt'}")
    print(f"  Missing in original     : {output_dir / 'unmatched_in_original.txt'}")

    if args.save_pair_metric_figures:
        pair_figure_dir = (
            Path(args.pair_figure_dir)
            if args.pair_figure_dir
            else output_dir / "pair_metric_figures"
        )
        print(f"  Pair figures            : {pair_figure_dir}")

    if args.save_edge_maps or args.save_edge_overlays:
        edge_map_dir = (
            Path(args.edge_map_dir)
            if args.edge_map_dir
            else output_dir / "edge_maps"
        )
        print(f"  Edge visualizations     : {edge_map_dir}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()