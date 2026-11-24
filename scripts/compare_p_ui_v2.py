#!/usr/bin/env python3
"""
Comparison experiment v2: Adaptive p_ui placement, adding Approach D
(histogram mode / pixel density peak).

Approaches at lux=100000:
    Current: p_ui = 0.70 + 0.10 * lux_strength
    A:       p_ui = stats["p50"]
    B:       p_ui = 0.6 * stats["p50"] + 0.4 * (0.70 + 0.10 * lux_strength)
    C:       p_ui = np.clip(stats["p50"], 0.45, 0.90)
    D:       p_ui = p_mode   (histogram mode / peak density bin center)
"""

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/wcm/sunlight_display/sunlight_enhance")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tone_curve import sunlight_curve, compute_luma_stats, apply_curve_y_only_bgr
from image_utils import compute_luma_float, smoothstep, image_to_rgb_float_for_plot
from image_io import imread_unicode, imwrite_unicode
from config import get_tone_params_by_lux


# ─── Paths ──────────────────────────────────────────────────────────────────────
IMG_PATH = Path("/mnt/data/wcm/sunlight_display/test_0603/中控/5.png")
OUT_DIR = Path("/mnt/data/wcm/sunlight_display/sunlight_enhance/output_images/_p_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Load image and compute Y ────────────────────────────────────────────────
img = imread_unicode(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(f"Cannot load image: {IMG_PATH}")

Y = compute_luma_float(img, standard="bt709").astype(np.float32)
Y_flat = Y.reshape(-1)


# ─── 2. Histogram mode (peak density bin center) ───────────────────────────────
hist, bin_edges = np.histogram(Y_flat, bins=128, range=(0.0, 1.0))
peak_bin = int(np.argmax(hist))
p_mode = float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)
print(f"Computed p_mode (histogram peak density): {p_mode:.4f}")


# ─── 3. Luma stats ─────────────────────────────────────────────────────────────
stats = compute_luma_stats(Y)
print(f"Image stats: p50={stats['p50']:.4f}, p_mode={p_mode:.4f}, "
      f"r_high={stats['r_high']:.4f}")


# ─── 4. Scoring (mirrors adapt_tone_params_by_content) ─────────────────────────
LUX = 100000
p, s = get_tone_params_by_lux(LUX)  # p=0.40, s=2.60
lux_strength = float(np.clip((s - 1.5) / 1.1, 0.0, 1.0))  # = 1.0

high_key_score = float(
    smoothstep(0.55, 0.80, stats["r_high"])
    * smoothstep(0.58, 0.72, stats["p50"])
)
high_key_p50 = float(smoothstep(0.60, 0.80, stats["p50"])) * 0.8
high_key_score = float(np.clip(max(high_key_score, high_key_p50), 0.0, 1.0))

dark_score = float(np.clip(
    smoothstep(0.45, 0.70, stats["r_low"])
    * (1.0 - smoothstep(0.32, 0.50, stats["p50"])),
    0.0, 1.0,
))

print(f"high_key_score={high_key_score:.4f}, dark_score={dark_score:.4f}, "
      f"lux_strength={lux_strength:.4f}")


# ─── p_ui per approach ─────────────────────────────────────────────────────────
p_ui_current = 0.70 + 0.10 * lux_strength                  # 0.80
p_ui_a = float(stats["p50"])
p_ui_b = 0.6 * float(stats["p50"]) + 0.4 * (0.70 + 0.10 * lux_strength)
p_ui_c = float(np.clip(stats["p50"], 0.45, 0.90))
p_ui_d = p_mode


def compute_p_adj(p_base, p_ui):
    """p_adj = p * (1 - hk) + p_ui * hk, then clip [0.25, 0.85].
    dark_score is 0 for this bright image, so the dark branch is skipped."""
    p_adj = p_base * (1.0 - high_key_score) + p_ui * high_key_score
    p_adj = p_adj * (1.0 - dark_score) + (0.30 + 0.08 * lux_strength) * dark_score
    return float(np.clip(p_adj, 0.25, 0.85))


p_adj_current = compute_p_adj(p, p_ui_current)
p_adj_a = compute_p_adj(p, p_ui_a)
p_adj_b = compute_p_adj(p, p_ui_b)
p_adj_c = compute_p_adj(p, p_ui_c)
p_adj_d = compute_p_adj(p, p_ui_d)


# ─── 5. s_high / s_low / curve_kwargs ───────────────────────────────────────────
s_scale_ui_high = 0.80 + 0.10 * lux_strength               # 0.90
s_scale_ui_low = 0.50 + 0.10 * lux_strength                # 0.60
s_scale_dark = 0.9 + 0.10 * lux_strength

s_high = s * (1.0 - high_key_score) + (s * s_scale_ui_high) * high_key_score
s_high = s_high * (1.0 - dark_score) + (s * s_scale_dark) * dark_score
s_high = float(np.clip(s_high, 1.2, 3.0))

s_low = s * (1.0 - high_key_score) + (s * s_scale_ui_low) * high_key_score
s_low = s_low * (1.0 - dark_score) + (s * s_scale_dark) * dark_score
s_low = float(np.clip(s_low, 1.2, 3.0))

curve_kwargs = {
    "s_low": s_low,
    "brightness_boost": 0.4 * lux_strength * dark_score,           # 0
    "brightness_start": 0.10,
    "brightness_end": 0.55,
    "high_protect": max(0.15 * lux_strength, 0.15 * high_key_score),
    "high_start": 0.55,
    "high_end": 1.0,
    "shadow_protect": 0.25 * dark_score,                            # 0
    "shadow_end": 0.3,
}

print(f"s_high={s_high:.4f}, s_low={s_low:.4f}")


# ─── 6. Apply tone curve and save outputs ──────────────────────────────────────
imwrite_unicode(str(OUT_DIR / "original.png"), img)

approaches = [
    ("current",    p_adj_current, p_ui_current, "Current (0.70+0.10ls)"),
    ("approach_a", p_adj_a,       p_ui_a,       "A: p50"),
    ("approach_b", p_adj_b,       p_ui_b,       "B: 0.6*p50+0.4*0.80"),
    ("approach_c", p_adj_c,       p_ui_c,       "C: clip(p50,0.45,0.90)"),
    ("approach_d", p_adj_d,       p_ui_d,       "D: p_mode"),
]

out_imgs = {}
for name, p_adj, p_ui, _label in approaches:
    out_bgr = apply_curve_y_only_bgr(
        img,
        p=p_adj,
        s=s_high,
        standard="bt709",
        curve_kwargs=curve_kwargs,
        lux=LUX,
    )
    imwrite_unicode(str(OUT_DIR / f"{name}.png"), out_bgr)
    out_imgs[name] = out_bgr
    print(f"  Saved: {name}.png  (p_ui={p_ui:.4f}, p_adj={p_adj:.4f})")


# ─── 7. Composite visualization ────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 10), dpi=200)
gs = GridSpec(
    2, 6,
    height_ratios=[1.0, 1.05],
    hspace=0.25,
    wspace=0.08,
    left=0.04, right=0.985, top=0.94, bottom=0.07,
)

# Top row: 6 images
top_panels = [
    ("Original",                         img,                 None,        None),
    (f"Current\np_ui=0.80, p_adj={p_adj_current:.3f}",  out_imgs["current"],    p_ui_current, p_adj_current),
    (f"A: p50\np_ui={p_ui_a:.3f}, p_adj={p_adj_a:.3f}", out_imgs["approach_a"], p_ui_a,       p_adj_a),
    (f"B: blend\np_ui={p_ui_b:.3f}, p_adj={p_adj_b:.3f}", out_imgs["approach_b"], p_ui_b,     p_adj_b),
    (f"C: clip\np_ui={p_ui_c:.3f}, p_adj={p_adj_c:.3f}",  out_imgs["approach_c"], p_ui_c,     p_adj_c),
    (f"D: p_mode\np_ui={p_ui_d:.3f}, p_adj={p_adj_d:.3f}", out_imgs["approach_d"], p_ui_d,    p_adj_d),
]

for col, (title, image, _pui, _padj) in enumerate(top_panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(image_to_rgb_float_for_plot(image))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# Bottom: histogram + tone curves
ax_bot = fig.add_subplot(gs[1, :])

# Y histogram (gray, alpha=0.3, normalized)
hist_norm = hist.astype(np.float32) / max(hist.max(), 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
ax_bot.bar(
    bin_centers, hist_norm,
    width=(bin_edges[1] - bin_edges[0]),
    color="gray", alpha=0.3, edgecolor="none",
    label="Y histogram (normalized)",
)

# y = x reference
xs = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
ax_bot.plot(xs, xs, "k--", linewidth=1.2, alpha=0.6, label="y = x")

# Tone curves
curve_specs = [
    ("Current", p_adj_current, "#1f77b4"),
    ("A: p50",  p_adj_a,       "#ff7f0e"),
    ("B: blend",p_adj_b,       "#2ca02c"),
    ("C: clip", p_adj_c,       "#9467bd"),
    ("D: mode", p_adj_d,       "#d62728"),
]

for label, p_adj, color in curve_specs:
    ys = sunlight_curve(xs, p=p_adj, s=s_high, **curve_kwargs)
    ax_bot.plot(xs, ys, color=color, linewidth=2.0,
                label=f"{label}  p_adj={p_adj:.3f}")
    ax_bot.scatter([p_adj], [p_adj], s=42, color=color, zorder=5,
                   edgecolors="white", linewidths=0.8)

# Vertical markers for p_mode and p50
ax_bot.axvline(p_mode, color="red", linestyle="--", linewidth=1.4,
               alpha=0.85, label=f"p_mode={p_mode:.3f}")
ax_bot.axvline(stats["p50"], color="blue", linestyle="--", linewidth=1.4,
               alpha=0.85, label=f"p50={stats['p50']:.3f}")

ax_bot.set_xlim(0.0, 1.0)
ax_bot.set_ylim(0.0, 1.05)
ax_bot.set_xlabel("Input luminance x")
ax_bot.set_ylabel("Output luminance y / normalized density")
ax_bot.set_title(
    f"Tone curves & Y histogram   "
    f"(lux={LUX}, s_high={s_high:.3f}, s_low={s_low:.3f}, "
    f"hk={high_key_score:.4f}, dark={dark_score:.2f})",
    fontsize=11,
)
ax_bot.grid(True, linestyle=":", alpha=0.45)
ax_bot.legend(loc="lower right", fontsize=9, framealpha=0.85, ncol=2)

fig.suptitle(
    f"Adaptive p_ui comparison (with Approach D = histogram mode)   "
    f"image: {IMG_PATH.name}",
    fontsize=13, y=0.985,
)

composite_path = OUT_DIR / "comparison_composite_v2.png"
fig.savefig(composite_path, dpi=200)
plt.close(fig)
print(f"\nComposite saved: {composite_path}")


# ─── 8. Comparison table ───────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"Image stats: p50={stats['p50']:.4f}, p_mode={p_mode:.4f}, "
      f"r_high={stats['r_high']:.4f}, high_key_score={high_key_score:.4f}")
print()
print(f"{LUX} lux comparison:")
print(f"  Current:   p_ui=0.80      → p_adj={p_adj_current:.4f}")
print(f"  A (p50):   p_ui={p_ui_a:.4f}    → p_adj={p_adj_a:.4f}")
print(f"  B (blend): p_ui={p_ui_b:.4f}    → p_adj={p_adj_b:.4f}")
print(f"  C (clip):  p_ui={p_ui_c:.4f}    → p_adj={p_adj_c:.4f}")
print(f"  D (mode):  p_ui={p_ui_d:.4f}    → p_adj={p_adj_d:.4f}")
print("=" * 72)
print("Done.")
