#!/usr/bin/env python3
"""
Compare four p_adj selection strategies (no base-p blending) at 100k lux:
    1. p_mode      — peak bin of the 128-bin Y histogram
    2. p50_direct  — median Y
    3. p_iqr       — (p25 + p75) / 2
    4. p_centroid  — weighted centroid of histogram bins whose density
                     exceeds 50% of the peak ("dense region")

All approaches share the same s_high / s_low / curve_kwargs so that only
the curve crossing point differs.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/wcm/sunlight_display/sunlight_enhance")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tone_curve import sunlight_curve, compute_luma_stats, apply_curve_y_only_bgr
from image_utils import compute_luma_float, image_to_rgb_float_for_plot


# ─── Paths ──────────────────────────────────────────────────────────────────────
IMG_PATH = Path("/mnt/data/wcm/sunlight_display/test_0603/中控/5.png")
OUT_DIR = Path(
    "/mnt/data/wcm/sunlight_display/sunlight_enhance/output_images/_p_comparison"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Load image & compute Y ──────────────────────────────────────────────────
img = cv2.imread(str(IMG_PATH), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Cannot load image: {IMG_PATH}")

Y = compute_luma_float(img, standard="bt709").astype(np.float32)
Y_flat = Y.reshape(-1)


# ─── 2. Compute p_mode / p50 / p_iqr / p_centroid ───────────────────────────────
N_BINS = 128
hist, bin_edges = np.histogram(Y_flat, bins=N_BINS, range=(0.0, 1.0))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

# p_mode — peak bin center
peak_bin = int(np.argmax(hist))
p_mode = float(bin_centers[peak_bin])

# percentile stats
stats = compute_luma_stats(Y)
p25 = float(stats["p25"])
p50 = float(stats["p50"])
p75 = float(stats["p75"])

# p_iqr — IQR center
p_iqr = float((p25 + p75) / 2.0)

# p_centroid — weighted centroid of dense region (bins > 50% of peak)
peak_count = int(hist.max())
dense_mask = hist > 0.5 * peak_count
dense_bin_centers = bin_centers[dense_mask]
dense_bin_counts = hist[dense_mask]
p_centroid = float(np.average(dense_bin_centers, weights=dense_bin_counts))


# ─── 3. Tone parameters (no blending with base p) ───────────────────────────────
LUX = 100000

# Fixed s parameters per requirements
s_high = 2.42
s_low = 1.879

curve_kwargs = {
    "s_low": s_low,
    "brightness_boost": 0.0,
    "brightness_start": 0.10,
    "brightness_end": 0.55,
    "high_protect": 0.15,
    "high_start": 0.55,
    "high_end": 1.0,
    "shadow_protect": 0.0,
    "shadow_end": 0.3,
}

# Approach-specific p_adj (no clipping/blending — direct values)
methods = [
    ("p_mode",     p_mode,     "p_mode.png",     "#d62728"),
    ("p50_direct", p50,        "p50_direct.png", "#2ca02c"),
    ("IQR center", p_iqr,      "p_iqr.png",      "#1f77b4"),
    ("Centroid",   p_centroid, "p_centroid.png", "#9467bd"),
]


# ─── 4. Apply curve & save per-method outputs ───────────────────────────────────
cv2.imwrite(str(OUT_DIR / "original.png"), img)

method_outputs = {}
for name, p_adj, fname, _color in methods:
    out_img = apply_curve_y_only_bgr(
        img, p=float(p_adj), s=s_high,
        standard="bt709", curve_kwargs=curve_kwargs, lux=LUX,
    )
    cv2.imwrite(str(OUT_DIR / fname), out_img)
    method_outputs[name] = out_img


# ─── 5. Composite visualization ─────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10), dpi=200)
gs = GridSpec(
    2, 5,
    height_ratios=[1.0, 1.05],
    hspace=0.20, wspace=0.04,
    left=0.04, right=0.99, top=0.93, bottom=0.075,
)

top_panels = [
    ("Original", img, None),
    (f"p_mode\np_adj={p_mode:.3f}",       method_outputs["p_mode"],     "#d62728"),
    (f"p50_direct\np_adj={p50:.3f}",      method_outputs["p50_direct"], "#2ca02c"),
    (f"IQR center\np_adj={p_iqr:.3f}",    method_outputs["IQR center"], "#1f77b4"),
    (f"Centroid\np_adj={p_centroid:.3f}", method_outputs["Centroid"],   "#9467bd"),
]
for col, (title, image, color) in enumerate(top_panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(image_to_rgb_float_for_plot(image))
    ax.set_title(title, fontsize=11,
                 color=color if color is not None else "black",
                 fontweight="bold" if color is not None else "normal")
    ax.set_xticks([])
    ax.set_yticks([])


ax_bot = fig.add_subplot(gs[1, :])

# Dense region shading (bins > 50% peak)
if dense_bin_centers.size > 0:
    bin_w = bin_edges[1] - bin_edges[0]
    dense_left = float(dense_bin_centers.min() - bin_w / 2.0)
    dense_right = float(dense_bin_centers.max() + bin_w / 2.0)
    ax_bot.axvspan(
        dense_left, dense_right,
        color="#fff3a8", alpha=0.55, zorder=0,
        label=f"Dense region (>50% peak): [{dense_left:.3f}, {dense_right:.3f}]",
    )

# Y histogram
hist_norm = hist.astype(np.float32) / max(hist.max(), 1)
ax_bot.bar(
    bin_centers, hist_norm,
    width=(bin_edges[1] - bin_edges[0]),
    color="gray", alpha=0.3, edgecolor="none",
    label="Y histogram (normalized, 128 bins)",
    zorder=1,
)

# Tone curves
xs = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
ax_bot.plot(xs, xs, "k--", linewidth=1.2, alpha=0.6, label="y = x", zorder=2)

for name, p_adj, _fname, color in methods:
    ys = sunlight_curve(xs, p=float(p_adj), s=s_high, **curve_kwargs)
    ax_bot.plot(
        xs, ys, color=color, linewidth=2.2,
        label=f"{name}  p_adj={p_adj:.3f}",
        zorder=3,
    )
    ax_bot.scatter(
        [p_adj], [p_adj], s=55, color=color, zorder=6,
        edgecolors="white", linewidths=1.0,
    )

# Vertical lines for each p_adj
for name, p_adj, _fname, color in methods:
    ax_bot.axvline(
        p_adj, color=color, linestyle="--", linewidth=1.3,
        alpha=0.85, zorder=4,
    )

ax_bot.set_xlim(0.0, 1.0)
ax_bot.set_ylim(0.0, 1.05)
ax_bot.set_xlabel("Input luminance x", fontsize=11)
ax_bot.set_ylabel("Output luminance y / normalized density", fontsize=11)
ax_bot.set_title(
    f"Tone curves vs Y histogram   "
    f"(lux={LUX}, s_high={s_high:.3f}, s_low={s_low:.3f})",
    fontsize=12,
)
ax_bot.grid(True, linestyle=":", alpha=0.45)
ax_bot.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)

fig.suptitle(
    f"p_adj Selection Methods Comparison  —  image: {IMG_PATH.name}",
    fontsize=14, y=0.985,
)

composite_path = OUT_DIR / "comparison_center_methods.png"
fig.savefig(composite_path, dpi=200)
plt.close(fig)


# ─── 6. Print results ───────────────────────────────────────────────────────────
print(f"p_mode = {p_mode:.4f}")
print(f"p50 = {p50:.4f}")
print(f"p25 = {p25:.4f}, p75 = {p75:.4f}")
print(f"p_iqr = (p25+p75)/2 = {p_iqr:.4f}")
print(f"p_centroid = {p_centroid:.4f} (weighted centroid of dense region)")
print()
print("Methods comparison at 100k lux:")
print(f"  p_mode:     p_adj = {p_mode:.4f}")
print(f"  p50_direct: p_adj = {p50:.4f}")
print(f"  IQR center: p_adj = {p_iqr:.4f}")
print(f"  Centroid:   p_adj = {p_centroid:.4f}")
print()
print(f"Saved to:  {OUT_DIR}")
print(f"Composite: {composite_path}")
