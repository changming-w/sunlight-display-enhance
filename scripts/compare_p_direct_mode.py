#!/usr/bin/env python3
"""
Comparison: direct p_adj = p_mode vs old blended approach (D).

The old approach blended the lux-preset base p with p_ui (= p_mode for
approach D):
    p_adj_old = p * (1 - high_key_score) + p_mode * high_key_score

The new approach lets each image use its own histogram mode directly:
    p_adj_new = p_mode

Rationale: each image's tone-curve crossing should be derived from its own
pixel distribution, independent of the lux-preset base p.
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


# ─── Paths ──────────────────────────────────────────────────────────────────────
IMG_PATH = Path("/mnt/data/wcm/sunlight_display/test_0603/中控/5.png")
OUT_DIR = Path(
    "/mnt/data/wcm/sunlight_display/sunlight_enhance/output_images/_p_comparison"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Load image ──────────────────────────────────────────────────────────────
img = cv2.imread(str(IMG_PATH), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Cannot load image: {IMG_PATH}")


# ─── 2. Compute Y, p_mode, stats ────────────────────────────────────────────────
Y = compute_luma_float(img, standard="bt709").astype(np.float32)
Y_flat = Y.reshape(-1)

hist, bin_edges = np.histogram(Y_flat, bins=128, range=(0.0, 1.0))
peak_bin = int(np.argmax(hist))
p_mode = float((bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0)

stats = compute_luma_stats(Y)
p50 = float(stats["p50"])

# high_key_score (mirrors adapt_tone_params_by_content)
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


# ─── 3. Lux preset & adapted s values ───────────────────────────────────────────
LUX = 100000
p_base = 0.40
s_base = 2.60
lux_strength = 1.0

# s_high / s_low: use the existing s_scale logic, unchanged
s_scale_ui_high = 0.80 + 0.10 * lux_strength       # 0.90
s_scale_ui_low = 0.50 + 0.10 * lux_strength        # 0.60
s_scale_dark = 0.9 + 0.10 * lux_strength

s_high = s_base * (1.0 - high_key_score) + (s_base * s_scale_ui_high) * high_key_score
s_high = s_high * (1.0 - dark_score) + (s_base * s_scale_dark) * dark_score
s_high = float(np.clip(s_high, 1.2, 3.0))

s_low = s_base * (1.0 - high_key_score) + (s_base * s_scale_ui_low) * high_key_score
s_low = s_low * (1.0 - dark_score) + (s_base * s_scale_dark) * dark_score
s_low = float(np.clip(s_low, 1.2, 3.0))

# p_adj — old blended (approach D) vs new direct p_mode
p_ui_old = p_mode  # in approach D, p_ui = p_mode
p_adj_old = p_base * (1.0 - high_key_score) + p_ui_old * high_key_score
p_adj_old = p_adj_old * (1.0 - dark_score) + (0.30 + 0.08 * lux_strength) * dark_score
p_adj_old = float(np.clip(p_adj_old, 0.25, 0.85))

p_adj_new = float(np.clip(p_mode, 0.25, 0.85))

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


# ─── 4. Apply tone curve & save outputs ─────────────────────────────────────────
cv2.imwrite(str(OUT_DIR / "original.png"), img)

out_old = apply_curve_y_only_bgr(
    img, p=p_adj_old, s=s_high,
    standard="bt709", curve_kwargs=curve_kwargs, lux=LUX,
)
cv2.imwrite(str(OUT_DIR / "old_blended.png"), out_old)

out_new = apply_curve_y_only_bgr(
    img, p=p_adj_new, s=s_high,
    standard="bt709", curve_kwargs=curve_kwargs, lux=LUX,
)
cv2.imwrite(str(OUT_DIR / "direct_pmode.png"), out_new)


# ─── 5. Composite visualization ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), dpi=200)
gs = GridSpec(
    2, 3,
    height_ratios=[1.0, 1.05],
    hspace=0.22,
    wspace=0.06,
    left=0.05, right=0.985, top=0.93, bottom=0.075,
)

top_panels = [
    ("Original", img),
    (f"Old Blended (D)\np_adj={p_adj_old:.3f}", out_old),
    (f"Direct p_mode\np_adj={p_adj_new:.3f}", out_new),
]
for col, (title, image) in enumerate(top_panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(image_to_rgb_float_for_plot(image))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

ax_bot = fig.add_subplot(gs[1, :])

# Y histogram
hist_norm = hist.astype(np.float32) / max(hist.max(), 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
ax_bot.bar(
    bin_centers, hist_norm,
    width=(bin_edges[1] - bin_edges[0]),
    color="gray", alpha=0.3, edgecolor="none",
    label="Y histogram (normalized)",
)

# Tone curves
xs = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
ax_bot.plot(xs, xs, "k--", linewidth=1.2, alpha=0.6, label="y = x")

curve_specs = [
    ("Old Blended",  p_adj_old, "#1f77b4"),
    ("Direct p_mode", p_adj_new, "#d62728"),
]
for label, p_adj, color in curve_specs:
    ys = sunlight_curve(xs, p=p_adj, s=s_high, **curve_kwargs)
    ax_bot.plot(xs, ys, color=color, linewidth=2.2,
                label=f"{label}  p_adj={p_adj:.3f}")
    ax_bot.scatter([p_adj], [p_adj], s=48, color=color, zorder=5,
                   edgecolors="white", linewidths=0.9)

# p_mode marker
ax_bot.axvline(p_mode, color="red", linestyle="--", linewidth=1.4,
               alpha=0.85, label=f"p_mode={p_mode:.3f}")

ax_bot.set_xlim(0.0, 1.0)
ax_bot.set_ylim(0.0, 1.05)
ax_bot.set_xlabel("Input luminance x")
ax_bot.set_ylabel("Output luminance y / normalized density")
ax_bot.set_title(
    f"Tone curves vs Y histogram   "
    f"(lux={LUX}, s_high={s_high:.3f}, s_low={s_low:.3f}, "
    f"hk={high_key_score:.4f})",
    fontsize=11,
)
ax_bot.grid(True, linestyle=":", alpha=0.45)
ax_bot.legend(loc="lower right", fontsize=9, framealpha=0.85, ncol=2)

fig.suptitle(
    f"Direct p_mode vs Old Blended  —  image: {IMG_PATH.name}",
    fontsize=13, y=0.985,
)

composite_path = OUT_DIR / "comparison_direct_pmode.png"
fig.savefig(composite_path, dpi=200)
plt.close(fig)


# ─── 6. Print results ───────────────────────────────────────────────────────────
print(f"p_mode = {p_mode:.4f}")
print(f"p50 = {p50:.4f}")
print(f"high_key_score = {high_key_score:.4f}")
print()
print(f"Old blended: p_adj = {p_adj_old:.3f} (p={p_base:.2f} blended with p_ui={p_mode:.2f})")
print(f"Direct mode: p_adj = p_mode = {p_adj_new:.4f} (no blending)")
print()
print(f"s_high={s_high:.4f}, s_low={s_low:.4f}")
print(f"high_protect={curve_kwargs['high_protect']:.4f}")
print(f"\nSaved to: {OUT_DIR}")
print(f"Composite: {composite_path}")
