#!/usr/bin/env python3
"""
Comparison experiment: 3 approaches for adaptive p_ui placement.

Approaches:
  Current: p_ui = 0.70 + 0.10 * lux_strength
  A:       p_ui = stats["p50"]
  B:       p_ui = 0.6 * stats["p50"] + 0.4 * (0.70 + 0.10 * lux_strength)
  C:       p_ui = np.clip(stats["p50"], 0.45, 0.90)
"""

import sys
from pathlib import Path

import numpy as np

# Ensure the project directory is on sys.path
sys.path.insert(0, "/mnt/data/wcm/sunlight_display/sunlight_enhance")

from config import get_tone_params_by_lux
from image_io import imread_unicode, imwrite_unicode
from image_utils import compute_luma_float, smoothstep
from tone_curve import (
    compute_luma_stats,
    extract_luma_for_tone_stats,
    apply_curve_y_only_bgr,
)


# ─── Load image ────────────────────────────────────────────────────────────────
IMG_PATH = Path("/mnt/data/wcm/sunlight_display/test_0603/中控/5.png")
OUT_DIR = Path("/mnt/data/wcm/sunlight_display/sunlight_enhance/output_images/_p_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img = imread_unicode(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Cannot load image: {IMG_PATH}")

# ─── Compute luma stats (same for all lux) ──────────────────────────────────────
Y = extract_luma_for_tone_stats(img, standard="bt709")
stats = compute_luma_stats(Y)

print("=" * 80)
print(f"Image: {IMG_PATH.name}")
print(f"  p5={stats['p5']:.4f}  p25={stats['p25']:.4f}  p50={stats['p50']:.4f}  "
      f"p75={stats['p75']:.4f}  p95={stats['p95']:.4f}")
print(f"  r_high={stats['r_high']:.4f}  r_very_high={stats['r_very_high']:.4f}  "
      f"r_low={stats['r_low']:.4f}  r_very_low={stats['r_very_low']:.4f}")
print("=" * 80)


# ─── Scoring logic (mirrors adapt_tone_params_by_content exactly) ───────────────
def compute_scores(stats, s):
    lux_strength = float(np.clip((s - 1.5) / 1.1, 0.0, 1.0))

    high_key_score = float(
        smoothstep(0.55, 0.80, stats["r_high"])
        * smoothstep(0.58, 0.72, stats["p50"])
    )
    # p50 auxiliary trigger
    high_key_p50 = float(smoothstep(0.60, 0.80, stats["p50"])) * 0.8
    high_key_score = float(np.clip(max(high_key_score, high_key_p50), 0.0, 1.0))

    dark_score = (
        smoothstep(0.45, 0.70, stats["r_low"])
        * (1.0 - smoothstep(0.32, 0.50, stats["p50"]))
    )
    dark_score = float(np.clip(dark_score, 0.0, 1.0))

    return lux_strength, high_key_score, dark_score


def compute_p_adj(p, p_ui, high_key_score, dark_score, lux_strength):
    """Compute final p_adj given a p_ui value."""
    p_dark = 0.30 + 0.08 * lux_strength
    p_adj = p * (1.0 - high_key_score) + p_ui * high_key_score
    p_adj = p_adj * (1.0 - dark_score) + p_dark * dark_score
    p_adj = float(np.clip(p_adj, 0.25, 0.85))
    return p_adj


def compute_s_high_and_kwargs(p, s, high_key_score, dark_score, lux_strength):
    """Compute s_high, s_low, and curve_kwargs (independent of p_ui)."""
    s_scale_ui_high = 0.80 + 0.10 * lux_strength
    s_scale_ui_low = 0.50 + 0.10 * lux_strength
    s_scale_dark = 0.9 + 0.10 * lux_strength

    s_high = s * (1.0 - high_key_score) + (s * s_scale_ui_high) * high_key_score
    s_high = s_high * (1.0 - dark_score) + (s * s_scale_dark) * dark_score

    s_low = s * (1.0 - high_key_score) + (s * s_scale_ui_low) * high_key_score
    s_low = s_low * (1.0 - dark_score) + (s * s_scale_dark) * dark_score

    s_high = float(np.clip(s_high, 1.2, 3.0))
    s_low = float(np.clip(s_low, 1.2, 3.0))

    curve_kwargs = {
        "s_low": s_low,
        "brightness_boost": 0.4 * lux_strength * dark_score,
        "brightness_start": 0.10,
        "brightness_end": 0.55,
        "high_protect": max(0.15 * lux_strength, 0.15 * high_key_score),
        "high_start": 0.55,
        "high_end": 1.0,
        "shadow_protect": 0.25 * dark_score,
        "shadow_end": 0.3,
    }
    return s_high, s_low, curve_kwargs


# ─── Main comparison table ──────────────────────────────────────────────────────
lux_list = [5000, 45000, 100000]

print(f"\n{'LUX':>8} | {'lux_str':>7} | {'hk_score':>8} | {'dk_score':>8} | "
      f"{'p_adj_Cur':>9} | {'p_adj_A':>7} | {'p_adj_B':>7} | {'p_adj_C':>7} | "
      f"{'s_high':>6} | {'s_low':>6}")
print("-" * 110)

results_100k = {}

for lux in lux_list:
    p, s = get_tone_params_by_lux(lux)
    lux_strength, high_key_score, dark_score = compute_scores(stats, s)

    # Current approach
    p_ui_current = 0.70 + 0.10 * lux_strength
    p_adj_current = compute_p_adj(p, p_ui_current, high_key_score, dark_score, lux_strength)

    # Approach A: direct median
    p_ui_a = stats["p50"]
    p_adj_a = compute_p_adj(p, p_ui_a, high_key_score, dark_score, lux_strength)

    # Approach B: weighted blend
    p_ui_b = 0.6 * stats["p50"] + 0.4 * (0.70 + 0.10 * lux_strength)
    p_adj_b = compute_p_adj(p, p_ui_b, high_key_score, dark_score, lux_strength)

    # Approach C: clipped median
    p_ui_c = float(np.clip(stats["p50"], 0.45, 0.90))
    p_adj_c = compute_p_adj(p, p_ui_c, high_key_score, dark_score, lux_strength)

    # s_high (same for all approaches)
    s_high, s_low, curve_kwargs = compute_s_high_and_kwargs(p, s, high_key_score, dark_score, lux_strength)

    print(f"{lux:>8} | {lux_strength:>7.4f} | {high_key_score:>8.4f} | {dark_score:>8.4f} | "
          f"{p_adj_current:>9.4f} | {p_adj_a:>7.4f} | {p_adj_b:>7.4f} | {p_adj_c:>7.4f} | "
          f"{s_high:>6.3f} | {s_low:>6.3f}")

    # Store 100k results for image output
    if lux == 100000:
        results_100k = {
            "p": p,
            "s_high": s_high,
            "curve_kwargs": curve_kwargs,
            "p_adj_current": p_adj_current,
            "p_adj_a": p_adj_a,
            "p_adj_b": p_adj_b,
            "p_adj_c": p_adj_c,
            "lux_strength": lux_strength,
            "high_key_score": high_key_score,
            "dark_score": dark_score,
        }

print("-" * 110)
print(f"\nImage stats:  p50={stats['p50']:.4f}  r_high={stats['r_high']:.4f}  "
      f"high_key_score(100k)={results_100k['high_key_score']:.4f}  "
      f"dark_score(100k)={results_100k['dark_score']:.4f}")


# ─── Generate comparison images at 100k lux ────────────────────────────────────
print("\n\nGenerating comparison images at lux=100000 ...")

# Save original
imwrite_unicode(str(OUT_DIR / "original.png"), img)
print(f"  Saved: original.png")

approaches = {
    "current": results_100k["p_adj_current"],
    "approach_a": results_100k["p_adj_a"],
    "approach_b": results_100k["p_adj_b"],
    "approach_c": results_100k["p_adj_c"],
}

s_high = results_100k["s_high"]
curve_kwargs = results_100k["curve_kwargs"]

for name, p_adj in approaches.items():
    out = apply_curve_y_only_bgr(
        img,
        p=p_adj,
        s=s_high,
        standard="bt709",
        curve_kwargs=curve_kwargs,
        lux=100000,
    )
    out_path = OUT_DIR / f"{name}.png"
    imwrite_unicode(str(out_path), out)
    print(f"  Saved: {name}.png  (p_adj={p_adj:.4f}, s_high={s_high:.3f})")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("\nDone.")
