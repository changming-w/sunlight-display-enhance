#!/usr/bin/env python3
"""
Composite visualization for p_ui comparison experiment.
Top row: 5 images (Original + 4 approaches)
Bottom panel: Y histogram + overlaid tone curves
"""

import sys
sys.path.insert(0, '/mnt/data/wcm/sunlight_display/sunlight_enhance')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tone_curve import sunlight_curve
from image_utils import compute_luma_float, smoothstep

# === Configuration ===
IMG_DIR = '/mnt/data/wcm/sunlight_display/sunlight_enhance/output_images/_p_comparison'
ORIG_IMAGE_PATH = '/mnt/data/wcm/sunlight_display/test_0603/中控/5.png'
SAVE_PATH = f'{IMG_DIR}/comparison_composite.png'

# Approach parameters
approaches = {
    'Current': {'p_adj': 0.6771, 'color': '#1f77b4'},
    'Approach A': {'p_adj': 0.6453, 'color': '#ff7f0e'},
    'Approach B': {'p_adj': 0.6580, 'color': '#2ca02c'},
    'Approach C': {'p_adj': 0.6453, 'color': '#d62728'},
}

# Shared parameters
s_high = 2.42
high_key_score = 0.6928
dark_score = 0.0
lux_strength = 1.0

# Compute curve_kwargs matching adapt_tone_params_by_content logic
curve_kwargs = {
    "s_low": 1.879,
    "brightness_boost": 0.4 * lux_strength * dark_score,
    "brightness_start": 0.10,
    "brightness_end": 0.55,
    "high_protect": max(0.15 * lux_strength, 0.15 * high_key_score),
    "high_start": 0.55,
    "high_end": 1.0,
    "shadow_protect": 0.25 * dark_score,
    "shadow_end": 0.3,
}

# Image filenames for top row
image_files = [
    ('Original', 'original.png', None),
    ('Current', 'current.png', 0.6771),
    ('Approach A', 'approach_a.png', 0.6453),
    ('Approach B', 'approach_b.png', 0.6580),
    ('Approach C', 'approach_c.png', 0.6453),
]

# === Create figure ===
fig = plt.figure(figsize=(20, 10), dpi=200, constrained_layout=True)

# Top row: 5 images
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.25, wspace=0.05)

for col, (name, fname, p_val) in enumerate(image_files):
    ax = fig.add_subplot(gs[0, col])
    img = plt.imread(f'{IMG_DIR}/{fname}')
    ax.imshow(img)
    ax.set_axis_off()
    if p_val is not None:
        ax.set_title(f'{name}\np_adj={p_val:.4f}', fontsize=10, fontweight='bold')
    else:
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')

# Bottom panel: single axis spanning all 5 columns
ax_bottom = fig.add_subplot(gs[1, :])

# --- Y histogram ---
img_bgr = cv2.imread(ORIG_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
if img_bgr is None:
    raise FileNotFoundError(f"Cannot load image: {ORIG_IMAGE_PATH}")
Y = compute_luma_float(img_bgr)
Y_flat = Y.ravel()

# Plot histogram (normalized, semi-transparent background)
counts, bin_edges = np.histogram(Y_flat, bins=100, range=(0.0, 1.0))
counts_norm = counts / counts.max()  # normalize to [0, 1]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax_bottom.fill_between(bin_centers, 0, counts_norm, alpha=0.25, color='gray', label='Y histogram (original)')
ax_bottom.plot(bin_centers, counts_norm, color='gray', alpha=0.5, linewidth=0.8)

# --- Reference line y=x ---
x = np.linspace(0.0, 1.0, 1001)
ax_bottom.plot(x, x, 'k--', linewidth=1.2, alpha=0.6, label='y = x')

# --- Tone curves ---
for name, params in approaches.items():
    p_val = params['p_adj']
    color = params['color']
    y = sunlight_curve(x, p=p_val, s=s_high, **curve_kwargs)
    ax_bottom.plot(x, y, linewidth=2.2, color=color, label=f'{name} (p_adj={p_val:.4f})')
    # Mark p_adj point on curve (where input = p_adj)
    y_at_p = sunlight_curve(np.array([p_val]), p=p_val, s=s_high, **curve_kwargs)[0]
    ax_bottom.scatter([p_val], [y_at_p], s=50, color=color, zorder=5, edgecolors='black', linewidths=0.6)

ax_bottom.set_xlim(0.0, 1.0)
ax_bottom.set_ylim(0.0, 1.0)
ax_bottom.set_xlabel('Input luminance (Y)', fontsize=11)
ax_bottom.set_ylabel('Output luminance / Histogram (normalized)', fontsize=11)
ax_bottom.set_title('Tone Curves with Y Histogram (100k lux, content-adaptive)', fontsize=12, fontweight='bold')
ax_bottom.legend(loc='lower right', fontsize=9)
ax_bottom.grid(True, linestyle=':', alpha=0.4)
ax_bottom.set_aspect('equal')

plt.savefig(SAVE_PATH, dpi=200, bbox_inches='tight')
plt.close()

print(f"Composite saved to: {SAVE_PATH}")
