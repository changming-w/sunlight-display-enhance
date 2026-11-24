#!/home/hp/miniforge3/envs/sunlight/bin/python
# -*- coding: utf-8 -*-
"""
compare_tone_adapt.py

Compare OLD high_key_score / dark_score tone adaptation logic vs the
NEW (current Formula B) brightness-score logic from
``instrument.tone_curve.adapt_tone_params_by_content``.

For each input image (LUX -> base p/s from ``get_tone_params_by_lux``):
  * apply OLD adapt
  * apply NEW (current) adapt
  * write a single composite figure to
        test/output7/tone_compare/{stem}_compare.png

Composite layout (2 rows, 3 cols):
    Top    : Original | OLD | NEW   (with parameter annotations)
    Bottom : single tone-curve plot spanning all 3 columns
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.utils.image_io import imread_unicode, collect_image_paths
from common.utils.image_utils import smoothstep, compute_luma_float
from common.algorithm.curve_math import sunlight_curve
from instrument.tone_curve import (
    compute_luma_stats,
    adapt_tone_params_by_y_channel,
    apply_curve_y_only_bgr,
)
from instrument.config import get_tone_params_by_lux


# ============================================================
# Constants
# ============================================================
LUX = 80000
P_METHOD = "iqr"

INPUT_DIR = Path("/mnt/data/wcm/sunlight_enhance/test/input")
OUT_DIR = Path("/mnt/data/wcm/sunlight_enhance/test/output7/tone_compare")


# ============================================================
# OLD adaptation logic (legacy high_key_score / dark_score)
# ============================================================
def old_adapt(Y, p, s, p_method="iqr"):
    stats = compute_luma_stats(Y)
    lux_strength = float(np.clip((s - 1.5) / 1.1, 0.0, 1.0))

    high_key_score = float(
        smoothstep(0.55, 0.80, stats["r_high"])
        * smoothstep(0.58, 0.72, stats["p50"])
    )
    high_key_p50 = float(smoothstep(0.60, 0.80, stats["p50"])) * 0.8
    high_key_score = float(np.clip(max(high_key_score, high_key_p50), 0.0, 1.0))

    dark_score = (
        smoothstep(0.45, 0.70, stats["r_low"])
        * (1.0 - smoothstep(0.32, 0.50, stats["p50"]))
    )
    dark_score = float(np.clip(dark_score, 0.0, 1.0))

    s_scale_ui_high = 0.80 + 0.10 * lux_strength
    s_scale_ui_low = 0.50 + 0.10 * lux_strength
    s_scale_dark = 0.8 + 0.10 * lux_strength

    if p_method == "iqr":
        p_adj = stats["p_iqr"]
    elif p_method == "mode":
        p_adj = stats["p_mode"]
    elif p_method == "median":
        p_adj = stats["p50"]
    elif p_method == "centroid":
        p_adj = stats["p_centroid"]
    else:
        p_adj = stats["p_iqr"]

    s_high = s * (1.0 - high_key_score) + (s * s_scale_ui_high) * high_key_score
    s_high = s_high * (1.0 - dark_score) + (s * s_scale_dark) * dark_score
    s_low = s * (1.0 - high_key_score) + (s * s_scale_ui_low) * high_key_score
    s_low = s_low * (1.0 - dark_score) + (s * s_scale_dark) * dark_score

    p_adj = float(np.clip(p_adj, 0.40, 0.80))
    s_high = float(np.clip(s_high, 1.2, 3.0))
    s_low = float(np.clip(s_low, 1.2, 3.0))

    curve_kwargs = {
        "s_low": s_low,
        "brightness_boost": 0.0,
        "brightness_start": 0.10,
        "brightness_end": 0.55,
        "high_protect": max(0.15 * lux_strength, 0.15 * high_key_score),
        "high_start": 0.75,
        "high_end": 1.0,
        "shadow_protect": 0.25 * dark_score,
        "shadow_end": 0.3,
    }
    return p_adj, s_high, curve_kwargs, stats, high_key_score, dark_score


# ============================================================
# Brightness score (Formula B with p15/p50/p75 skew correction)
#   raw = p50 + 0.3 * ((p75 - p50) - (p50 - p15))
#   then dead-zone mapped to [-1, +1] over [0.15, 0.30] / [0.55, 0.80]
# ============================================================
def _deadzone_map(raw: float) -> float:
    if raw <= 0.15:
        return -1.0
    if raw < 0.30:
        return -1.0 + (raw - 0.15) / 0.15
    if raw <= 0.55:
        return 0.0
    if raw < 0.80:
        return (raw - 0.55) / 0.25
    return 1.0


def compute_brightness_b(p15: float, p50: float, p75: float) -> tuple:
    skew = (p75 - p50) - (p50 - p15)
    raw = p50 + 0.3 * skew
    return raw, _deadzone_map(raw)


# ============================================================
# Composite figure: 2 rows x 3 cols, bottom row spans all cols
# ============================================================
def make_composite(image_stem, img_bgr, Y, stats, p15, b_raw, b_score,
                   old_p, old_s, old_kwargs,
                   new_p, new_s, new_kwargs,
                   out_old, out_new,
                   base_p, base_s,
                   save_path):
    p50 = float(stats["p50"])
    p75 = float(stats["p75"])

    # 18 wide x 10 tall, top row taller than bottom curve
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3.0, 2.0],
                  hspace=0.18, wspace=0.04,
                  left=0.04, right=0.99, top=0.94, bottom=0.07)

    images = [img_bgr, out_old, out_new]
    titles = ["Original", "OLD", "NEW (Formula B)"]
    annotations = [
        f"p15={p15:.3f}  p50={p50:.3f}  p75={p75:.3f}",
        (f"p_adj={old_p:.3f}  s_high={old_s:.3f}  "
         f"s_low={old_kwargs['s_low']:.3f}"),
        (f"b={b_score:+.2f}  raw={b_raw:.3f}\n"
         f"p_adj={new_p:.3f}  s_high={new_s:.3f}  "
         f"s_low={new_kwargs['s_low']:.3f}"),
    ]

    for col, (im, title, ann) in enumerate(zip(images, titles, annotations)):
        ax = fig.add_subplot(gs[0, col])
        if im.dtype == np.uint8:
            disp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            disp = cv2.cvtColor(
                (np.clip(im, 0, 1) * 255).astype(np.uint8),
                cv2.COLOR_BGR2RGB,
            )
        ax.imshow(disp, aspect="equal")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(ann, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom: full-width curve plot
    ax_curve = fig.add_subplot(gs[1, :])

    x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    y_old = sunlight_curve(x, p=old_p, s=old_s, **old_kwargs)
    y_new = sunlight_curve(x, p=new_p, s=new_s, **new_kwargs)

    # Histogram background
    flat = Y.reshape(-1)
    if flat.size > 1_000_000:
        flat = flat[::max(1, flat.size // 1_000_000)]
    hist, edges = np.histogram(flat, bins=128, range=(0.0, 1.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if hist.max() > 0:
        hist_norm = hist / hist.max() * 0.45
    else:
        hist_norm = hist
    ax_curve.bar(
        centers, hist_norm, width=(edges[1] - edges[0]),
        color="#cccccc", alpha=0.55, edgecolor="none",
        label="Y histogram (orig)", zorder=1,
    )

    # Percentile markers
    ax_curve.axvline(p15, color="#7fbf7f", linestyle=":", linewidth=1.0,
                     label=f"p15={p15:.3f}", zorder=2)
    ax_curve.axvline(p50, color="#888888", linestyle=":", linewidth=1.0,
                     label=f"p50={p50:.3f}", zorder=2)
    ax_curve.axvline(p75, color="#bf7f7f", linestyle=":", linewidth=1.0,
                     label=f"p75={p75:.3f}", zorder=2)

    # y = x reference
    ax_curve.plot(x, x, color="#999999", linestyle="--", linewidth=1.0,
                  label="y = x", zorder=3)

    # OLD (red) and NEW (blue)
    ax_curve.plot(
        x, y_old, color="#d62728", linewidth=2.2, zorder=5,
        label=(f"OLD   p={old_p:.3f}  s_high={old_s:.3f}  "
               f"s_low={old_kwargs['s_low']:.3f}"),
    )
    ax_curve.plot(
        x, y_new, color="#1f77b4", linewidth=2.2, zorder=6,
        label=(f"NEW (Formula B)   b={b_score:+.2f}  p={new_p:.3f}  "
               f"s_high={new_s:.3f}  s_low={new_kwargs['s_low']:.3f}"),
    )

    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_ylim(0.0, 1.0)
    # NOTE: do NOT force equal aspect — we want the curve plot to stretch
    # to the full width of the top image row.
    ax_curve.set_xlabel("input Y", fontsize=10)
    ax_curve.set_ylabel("output Y", fontsize=10)
    ax_curve.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax_curve.set_title(
        f"{image_stem}    LUX={LUX}    "
        f"base p={base_p:.3f}, base s={base_s:.3f}",
        fontsize=11,
    )
    ax_curve.legend(loc="lower right", fontsize=9, framealpha=0.85)

    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_p, base_s = get_tone_params_by_lux(LUX)

    image_paths = collect_image_paths(INPUT_DIR, recursive=False)
    if not image_paths:
        print(f"[!] No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images. LUX={LUX}, "
          f"base p={base_p:.3f}, base s={base_s:.3f}, p_method={P_METHOD}")
    print()

    header = (
        f"{'filename':<24} {'p15':>5} {'p50':>5} {'p75':>5} | "
        f"{'OLD p':>6} {'OLD sH':>7} {'OLD sL':>7} | "
        f"{'NEW p':>6} {'NEW sH':>7} {'NEW sL':>7} | "
        f"{'b':>5}"
    )
    print(header)
    print("-" * len(header))

    count = 0

    for path in image_paths:
        img = imread_unicode(path)
        if img is None:
            print(f"[!] Failed to read: {path}")
            continue

        if img.ndim == 3 and img.shape[2] == 4:
            img_bgr = img[..., :3]
        else:
            img_bgr = img

        # Y in [0, 1] (BGR input, BT.709)
        Y = compute_luma_float(img_bgr, standard="bt709").astype(np.float32)

        stats = compute_luma_stats(Y)
        p50 = float(stats["p50"])
        p75 = float(stats["p75"])
        p15 = float(np.percentile(Y.reshape(-1), 15))

        # OLD
        old_p, old_s, old_kwargs, _, _, _ = old_adapt(
            Y, base_p, base_s, p_method=P_METHOD,
        )

        # NEW (current Formula B in adapt_tone_params_by_content)
        new_p, new_s, new_kwargs, _ = adapt_tone_params_by_y_channel(
            Y, base_p, base_s, p_method=P_METHOD,
        )

        # Brightness score (Formula B) — for annotation/diagnostic only
        b_raw, b_score = compute_brightness_b(p15, p50, p75)

        # Apply each
        out_old = apply_curve_y_only_bgr(
            img_bgr, p=old_p, s=old_s,
            standard="bt709", curve_kwargs=old_kwargs, lux=LUX,
        )
        out_new = apply_curve_y_only_bgr(
            img_bgr, p=new_p, s=new_s,
            standard="bt709", curve_kwargs=new_kwargs, lux=LUX,
        )

        # Composite
        save_path = OUT_DIR / f"{path.stem}_compare.png"
        make_composite(
            path.stem, img_bgr, Y, stats, p15, b_raw, b_score,
            old_p, old_s, old_kwargs,
            new_p, new_s, new_kwargs,
            out_old, out_new,
            base_p, base_s,
            save_path,
        )

        row = (
            f"{path.name:<24} {p15:>5.3f} {p50:>5.3f} {p75:>5.3f} | "
            f"{old_p:>6.3f} {old_s:>7.3f} {old_kwargs['s_low']:>7.3f} | "
            f"{new_p:>6.3f} {new_s:>7.3f} {new_kwargs['s_low']:>7.3f} | "
            f"{b_score:>+5.2f}"
        )
        print(row)
        count += 1

    print()
    print(f"Composites saved -> {OUT_DIR}")
    print(f"Total processed: {count}")


if __name__ == "__main__":
    main()
