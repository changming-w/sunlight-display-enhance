#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Local RMS Contrast and EME for original/enhanced image folders.

Metrics
-------
1. Local RMS contrast on luminance channel:

    C(x) = local_std(Y) / (local_mean(Y) + eps)

2. EME, Enhancement Measure Estimation:

    EME = mean_blocks 20 * log10((I_high + eps) / (I_low + eps))

   classic mode:
        I_high = max(block), I_low = min(block)

   percentile mode:
        I_high = P95(block), I_low = P5(block), configurable.

Outputs
-------
output/
    original_metrics.csv
    enhanced_metrics.csv
    paired_comparison_metrics.csv
    unmatched_in_original.txt
    unmatched_in_enhanced.txt

Usage
-----
python scripts/compute_rms_eme_pair.py \
    --original ./test/original \
    --enhanced ./test/enhanced \
    --output ./test/rms_eme_eval \
    --rms-window 15 \
    --rms-filter gaussian \
    --rms-valid-min-mean 0.02 \
    --eme-mode both \
    --block-size 16

Notes
-----
- RMS contrast is a linear ratio-like value.
- EME is in dB.
- For enhancement comparison:
    rms_median_gain > 1 means local RMS contrast increases.
    eme_percentile_delta_db > 0 means percentile EME increases.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# -----------------------------
# IO utilities
# -----------------------------

def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Read image with Unicode path support."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def collect_image_paths(input_path: str, recursive: bool = False) -> List[Path]:
    """Collect image paths from file or folder."""
    p = Path(input_path)

    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTS:
            return [p]
        return []

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


def build_image_dict(
    folder: str,
    recursive: bool = False,
    match_mode: str = "basename"
) -> Tuple[Dict[str, Path], List[str]]:
    """
    Build key -> image path dictionary.

    match_mode:
        basename:
            use image filename as key, e.g. a.png
        relative:
            use relative path as key, e.g. scene1/a.png
    """
    root = Path(folder)
    paths = collect_image_paths(folder, recursive=recursive)

    image_dict: Dict[str, Path] = {}
    duplicate_keys: List[str] = []

    if root.is_file():
        base_dir = root.parent
    else:
        base_dir = root

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


def write_csv(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    preferred_fields: Optional[List[str]] = None
) -> None:
    """Write rows to CSV with UTF-8 BOM for Excel compatibility."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames: List[str] = []

    if preferred_fields is not None:
        for k in preferred_fields:
            if k in all_keys and k not in fieldnames:
                fieldnames.append(k)

    rest = sorted(list(all_keys - set(fieldnames)))
    fieldnames.extend(rest)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text_list(path: Path, items: List[str]) -> None:
    """Write a list of strings into txt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(str(item) + "\n")


# -----------------------------
# Image conversion
# -----------------------------

def normalize_to_float01(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to float32 [0, 1].

    Integer images are normalized by dtype max.
    Float images larger than 1 are heuristically normalized.
    """
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
    """Inverse sRGB transfer function."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)

    y = np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4)
    )

    return y.astype(np.float32)


def to_luminance_float01(
    img: np.ndarray,
    luma: str = "bt601",
    linearize_srgb: bool = False
) -> np.ndarray:
    """
    Convert image to luminance/luma float32 [0, 1].

    OpenCV color image order is BGR.

    luma:
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

    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


# -----------------------------
# Local RMS contrast
# -----------------------------

def ensure_odd_kernel(k: int) -> int:
    """Ensure positive odd kernel size."""
    k = int(k)

    if k < 3:
        k = 3

    if k % 2 == 0:
        k += 1

    return k


def compute_local_rms_contrast(
    y: np.ndarray,
    window: int = 15,
    filter_type: str = "gaussian",
    sigma: float = 0.0,
    eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local RMS contrast on luminance image.

    C(x) = local_std(Y) / (local_mean(Y) + eps)

    Args:
        y:
            Luminance image, float32 [0, 1].
        window:
            Local window size.
        filter_type:
            gaussian or box.
        sigma:
            Gaussian sigma. If 0, OpenCV chooses automatically.
        eps:
            Denominator epsilon.

    Returns:
        contrast:
            Local RMS contrast map.
        mean:
            Local mean map.
        std:
            Local std map.
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

    return (
        contrast.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32)
    )


def summarize_contrast_map(
    contrast: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Summarize local RMS contrast map."""
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


# -----------------------------
# EME
# -----------------------------

def compute_eme_scores(
    y: np.ndarray,
    block_h: int = 16,
    block_w: int = 16,
    mode: str = "classic",
    p_low: float = 5.0,
    p_high: float = 95.0,
    eps: float = 1.0 / 255.0,
    drop_incomplete: bool = False
) -> np.ndarray:
    """
    Compute block-level EME scores.

    Args:
        y:
            Luminance image, float32 [0, 1].
        block_h:
            Block height.
        block_w:
            Block width.
        mode:
            classic or percentile.
        p_low:
            Low percentile for percentile EME.
        p_high:
            High percentile for percentile EME.
        eps:
            Epsilon for ratio.
        drop_incomplete:
            Whether to drop incomplete border blocks.

    Returns:
        scores:
            Block-level EME scores in dB.
    """
    if mode not in ["classic", "percentile"]:
        raise ValueError(f"Unknown EME mode: {mode}")

    y = y.astype(np.float32)

    h, w = y.shape

    block_h = max(1, int(block_h))
    block_w = max(1, int(block_w))

    scores: List[float] = []

    for y0 in range(0, h, block_h):
        y1 = min(y0 + block_h, h)

        if drop_incomplete and (y1 - y0) < block_h:
            continue

        for x0 in range(0, w, block_w):
            x1 = min(x0 + block_w, w)

            if drop_incomplete and (x1 - x0) < block_w:
                continue

            block = y[y0:y1, x0:x1]

            if block.size == 0:
                continue

            if mode == "classic":
                low = float(np.min(block))
                high = float(np.max(block))
            else:
                low = float(np.percentile(block, p_low))
                high = float(np.percentile(block, p_high))

            if high < low:
                high, low = low, high

            ratio = (high + eps) / (low + eps)
            ratio = max(ratio, 1.0)

            score = 20.0 * np.log10(ratio)
            scores.append(float(score))

    # Fallback if all border blocks were dropped.
    if len(scores) == 0:
        if mode == "classic":
            low = float(np.min(y))
            high = float(np.max(y))
       :
            low = float(np.percentile(y, p_low))
            high = float(np.percentile(y, p_high))

        ratio = (high + eps) / (low + eps)
        ratio = max(ratio, 1.0)

        score = 20.0 * np.log10(ratio)
        scores.append(float(score))

    return np.asarray(scores, dtype=np.float32)


def summarize_eme_scores(
    scores: np.ndarray,
    mode: str
) -> Dict[str, float]:
    """Summarize EME block scores."""
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


# -----------------------------
# Metric computation
# -----------------------------

def get_eme_modes(eme_mode: str) -> List[str]:
    """Return EME modes to run."""
    if eme_mode == "both":
        return ["classic", "percentile"]

    return [eme_mode]


def compute_metrics_for_image(
    image_path: Path,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Compute RMS contrast and EME metrics for one image."""
    img = imread_unicode(str(image_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    y = to_luminance_float01(
        img,
        luma=args.luma,
        linearize_srgb=args.linearize_srgb
    )

    h, w = y.shape

    metrics: Dict[str, Any] = {
        "height": int(h),
        "width": int(w),
    }

    # Local RMS contrast.
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

    rms_stats = summarize_contrast_map(rms_map, mask=valid_mask)
    metrics.update(rms_stats)

    # EME.
    block_h = args.block_h if args.block_h is not None else args.block_size
    block_w = args.block_w if args.block_w is not None else args.block_size

    for mode in get_eme_modes(args.eme_mode):
        scores = compute_eme_scores(
            y,
            block_h=block_h,
            block_w=block_w,
            mode=mode,
            p_low=args.p_low,
            p_high=args.p_high,
            eps=args.eme_eps,
            drop_incomplete=args.drop_incomplete
        )

        eme_stats = summarize_eme_scores(scores, mode=mode)
        metrics.update(eme_stats)

    return metrics


def build_param_info(args: argparse.Namespace) -> Dict[str, Any]:
    """Build parameter info saved into CSV."""
    block_h = args.block_h if args.block_h is not None else args.block_size
    block_w = args.block_w if args.block_w is not None else args.block_size

    return {
        "luma": args.luma,
        "linearize_srgb": int(args.linearize_srgb),

        "rms_window": ensure_odd_kernel(args.rms_window),
        "rms_filter": args.rms_filter,
        "rms_sigma": args.rms_sigma,
        "rms_eps": args.rms_eps,
        "rms_valid_min_mean": args.rms_valid_min_mean,

        "eme_mode": args.eme_mode,
        "eme_block_h": block_h,
        "eme_block_w": block_w,
        "eme_eps": args.eme_eps,
        "eme_p_low": args.p_low,
        "eme_p_high": args.p_high,
        "eme_drop_incomplete": int(args.drop_incomplete),
    }


def finite_float(x: Any) -> Optional[float]:
    """Convert x to finite float if possible."""
    try:
        v = float(x)
    except Exception:
        return None

    if not np.isfinite(v):
        return None

    return v


def safe_delta(enh: Any, orig: Any) -> Any:
    """Compute enhanced - original safely."""
    e = finite_float(enh)
    o = finite_float(orig)

    if e is None or o is None:
        return ""

    return e - o


def safe_ratio(enh: Any, orig: Any, eps: float = 1e-12) -> Any:
    """Compute enhanced / original safely."""
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
    original_metrics: Dict[str, Any],
    enhanced_metrics: Dict[str, Any],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Build paired comparison row."""
    row: Dict[str, Any] = {
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

    # Save original/enhanced raw metric values.
    for k in metric_keys:
        row[f"orig_{k}"] = original_metrics.get(k, "")
        row[f"enh_{k}"] = enhanced_metrics.get(k, "")

    # RMS contrast comparison.
    rms_gain_keys = [
        "rms_mean",
        "rms_map_std",
        "rms_median",
        "rms_p75",
        "rms_p90",
        "rms_p95",
        "rms_max",
    ]

    for k in rms_gain_keys:
        if k in original_metrics or k in enhanced_metrics:
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

    # EME comparison.
    # EME is in dB, so delta_db is more meaningful than ratio.
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
            if k in original_metrics or k in enhanced_metrics:
                row[f"{k}_delta_db"] = safe_delta(
                    enhanced_metrics.get(k, ""),
                    original_metrics.get(k, "")
                )

    return row


def get_single_csv_preferred_fields(args: argparse.Namespace) -> List[str]:
    """Preferred field order for original/enhanced CSV."""
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
        "eme_block_h",
        "eme_block_w",
        "eme_eps",
        "eme_p_low",
        "eme_p_high",
        "eme_drop_incomplete",
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

    return fields


def get_comparison_csv_preferred_fields(args: argparse.Namespace) -> List[str]:
    """Preferred field order for paired comparison CSV."""
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

        "eme_mode",
        "eme_block_h",
        "eme_block_w",
        "eme_eps",
        "eme_p_low",
        "eme_p_high",
        "eme_drop_incomplete",
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

    return fields


def collect_numeric(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    """Collect finite numeric values from rows."""
    vals: List[float] = []

    for row in rows:
        v = finite_float(row.get(key, ""))

        if v is not None:
            vals.append(v)

    return np.asarray(vals, dtype=np.float64)


def print_summary(comparison_rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Print concise dataset-level summary."""
    print("-" * 80)
    print("Dataset Comparison Summary")
    print("-" * 80)

    for key in ["rms_mean_gain", "rms_median_gain", "rms_p90_gain"]:
        vals = collect_numeric(comparison_rows, key)

        if vals.size > 0:
            print(f"{key:32s}: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}")

    for mode in get_eme_modes(args.eme_mode):
        key = f"eme_{mode}_delta_db"
        vals = collect_numeric(comparison_rows, key)

        if vals.size > 0:
            print(f"{key:32s}: mean={np.mean(vals):.4f} dB, median={np.median(vals):.4f} dB")

    print("-" * 80)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute local RMS contrast and EME for original/enhanced image folders."
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
        help="Output folder. Default: ./test/rms_eme_eval"
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
        help=(
            "Image matching mode. "
            "basename: match by filename. "
            "relative: match by relative path. "
            "Default: basename."
        )
    )

    # Luminance options.
    parser.add_argument(
        "--luma",
        type=str,
        default="bt601",
        choices=["bt601", "bt709"],
        help="Luminance/luma conversion. Default: bt601."
    )
    parser.add_argument(
        "--linearize-srgb",
        action="store_true",
        help="Apply inverse sRGB transfer before luminance conversion."
    )

    # RMS options.
    parser.add_argument(
        "--rms-window",
        type=int,
        default=15,
        help="Local RMS window size. Will be forced to odd. Default: 15."
    )
    parser.add_argument(
        "--rms-filter",
        type=str,
        default="gaussian",
        choices=["gaussian", "box"],
        help="Local mean/std filter type. Default: gaussian."
    )
    parser.add_argument(
        "--rms-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for RMS. 0 lets OpenCV choose automatically. Default: 0."
    )
    parser.add_argument(
        "--rms-eps",
        type=float,
        default=1e-4,
        help="Epsilon in local RMS denominator. Default: 1e-4."
    )
    parser.add_argument(
        "--rms-valid-min-mean",
        type=float,
        default=0.0,
        help=(
            "Exclude pixels whose local mean is below this threshold. "
            "Useful for suppressing unstable dark-region RMS contrast. "
            "Default: 0."
        )
    )

    # EME options.
    parser.add_argument(
        "--eme-mode",
        type=str,
        default="both",
        choices=["classic", "percentile", "both"],
        help="EME mode. Default: both."
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Default square block size for EME. Default: 16."
    )
    parser.add_argument(
        "--block-h",
        type=int,
        default=None,
        help="EME block height. Overrides --block-size."
    )
    parser.add_argument(
        "--block-w",
        type=int,
        default=None,
        help="EME block width. Overrides --block-size."
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=5.0,
        help="Low percentile for percentile EME. Default: 5."
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=95.0,
        help="High percentile for percentile EME. Default: 95."
    )
    parser.add_argument(
        "--eme-eps",
        type=float,
        default=1.0 / 255.0,
        help="Epsilon for EME ratio. Default: 1/255."
    )
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Drop incomplete border blocks for EME."
    )

    args = parser.parse_args()

    if args.p_high <= args.p_low:
        print("[Error] --p-high must be larger than --p-low")
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
        print"[Warning] Duplicate keys in original folder: {len(original_duplicates)}")
        write_text_list(output_dir / "duplicate_keys_original.txt", sorted(original_duplicates))

    if enhanced_duplicates:
        print(f"[Warning] Duplicate keys in enhanced folder: {len(enhanced_duplicates)}")
        write_text_list(output_dir / "duplicate_keys_enhanced.txt", sorted(enhanced_duplicates))

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
    print("Local RMS Contrast + EME Pair Evaluation")
    print("=" * 80)
    print(f"Original folder      : {args.input_original}")
    print(f"Enhanced folder      : {args.input_enhanced}")
    print(f"Output               : {args.output}")
    print(f"Match mode           : {args.match_mode}")
    print(f"Original images      : {len(original_dict)}")
    print(f"Enhanced images      : {len(enhanced_dict)}")
    print(f"Matched pairs        : {len(common_keys)}")
    print(f"Unmatched original   : {len(missing_in_enhanced)}")
    print(f"Unmatched enhanced   : {len(missing_in_original)}")
    print("-" * 80)
    print(f"Luma                 : {args.luma}")
    print(f"Linearize sRGB       : {args.linearize_srgb}")
    print(f"RMS window/filter    : {ensure_odd_kernel(args.rms_window)} / {args.rms_filter}")
    print(f"RMS eps              : {args.rms_eps}")
    print(f"RMS valid min mean   : {args.rms_valid_min_mean}")
    print(f"EME mode             : {args.eme_mode}")
    print(f"EME block            : {param_info['eme_block_h']} x {param_info['eme_block_w']}")
    print(f"EME eps              : {args.eme_eps}")
    if args.eme_mode in ["percentile", "both"]:
        print(f"Percentile EME       : P{args.p_low} / P{args.p_high}")
    print("=" * 80)

    original_rows: List[Dict[str, Any]] = []
    enhanced_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

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

        original_row: Dict[str, Any] = {
            "key": key,
            "image": original_path.name,
            "path": str(original_path),
        }
        original_row.update(param_info)
        original_row.update(original_metrics)

        enhanced_row: Dict[str, Any] = {
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

        msg = (
            f"  RMS median: "
            f"{original_metrics.get('rms_median', 0):.6f} -> "
            f"{enhanced_metrics.get('rms_median', 0):.6f}"
        )

        rms_gain = safe_ratio(
            enhanced_metrics.get("rms_median", ""),
            original_metrics.get("rms_median", "")
        )

        if rms_gain != "":
            msg += f", gain={rms_gain:.4f}x"

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

        print(msg)

    original_csv = output_dir / "original_metrics.csv"
    enhanced_csv = output_dir / "enhanced_metrics.csv"
    comparison_csv = output_dir / "paired_comparison_metrics.csv"

    write_csv(
        original_csv,
        original_rows,
        preferred_fields=get_single_csv_preferred_fields(args)
    )

    write_csv(
        enhanced_csv,
        enhanced_rows,
        preferred_fields=get_single_csv_preferred_fields(args)
    )

    write_csv(
        comparison_csv,
        comparison_rows,
        preferred_fields=get_comparison_csv_preferred_fields(args)
    )

    print_summary(comparison_rows, args)

    print("Saved:")
    print(f"  Original metrics   : {original_csv}")
    print(f"  Enhanced metrics   : {enhanced_csv}")
    print(f"  Paired comparison  : {comparison_csv}")
    print(f"  Unmatched original : {output_dir / 'unmatched_in_enhanced.txt'}")
    print(f"  Unmatched enhanced : {output_dir / 'unmatched_in_original.txt'}")
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()