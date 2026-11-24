#!/usr/bin/env python3
"""
Batch comparison of four p_adj selection strategies (no base-p blending) at
100k lux, applied to every image inside /mnt/data/wcm/sunlight_display/test_0603/仪表/.

Methods:
    1. p_mode      — peak bin of the 128-bin Y histogram
    2. p50_direct  — median Y
    3. p_iqr       — (p25 + p75) / 2
    4. p_centroid  — weighted centroid of histogram bins whose density
                     exceeds 50% of the peak ("dense region")

p_adj values are clipped to [0.25, 0.85] before being fed to the curve.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/wcm/sunlight_display/sunlight_enhance")

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tone_curve import sunlight_curve, compute_luma_stats, apply_curve_y_only_bgr
from image_utils import compute_luma_float, smoothstep, image_to_rgb_float_for_plot


# ─── Common configuration ──────────────────────────────────────────────────────
LUX = 100000
BASE_P = 0.40
BASE_S = 2.60
LUX_STRENGTH = 1.0

S_SCALE_UI_HIGH = 0.90
S_SCALE_UI_LOW = 0.60
S_SCALE_DARK = 1.0
S_CLIP_LO = 1.2
S_CLIP_HI = 3.0

P_CLIP_LO = 0.25
P_CLIP_HI = 0.85

INPUT_DIR = Path("/mnt/data/wcm/sunlight_display/test_0603/仪表")
OUT_ROOT = Path(
    "/mnt/data/wcm/sunlight_display/sunlight_enhance/output_images/_p_comparison_batch_yibiao"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def imread_safe(path: Path):
    """Imread that handles unicode / Chinese filenames safely."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_safe(path: Path, img) -> bool:
    """Imwrite that handles unicode / Chinese filenames safely."""
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def process_image(img_path: Path) -> dict | None:
    stem = img_path.stem
    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image and compute Y
    img = imread_safe(img_path)
    if img is None:
        print(f"[WARN] cannot load {img_path}")
        return None

    Y = compute_luma_float(img, standard="bt709").astype(np.float32)

    # 2. Histogram-based statistics
    N_BINS = 128
    hist, bin_edges = np.histogram(Y.reshape(-1), bins=N_BINS, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peak_bin = int(np.argmax(hist))
    p_mode_raw = float(bin_centers[peak_bin])

    stats = compute_luma_stats(Y)
    p25 = float(stats["p25"])
    p50 = float(stats["p50"])
    p75 = float(stats["p75"])
    p_iqr_raw = float((p25 + p75) / 2.0)

    peak_count = int(hist.max())
    dense_mask = hist > 0.5 * peak_count
    dense_bin_centers = bin_centers[dense_mask]
    dense_bin_counts = hist[dense_mask]
    if dense_bin_centers.size > 0:
        p_centroid_raw = float(
            np.average(dense_bin_centers, weights=dense_bin_counts)
        )
    else:
        p_centroid_raw = p_mode_raw

    # 3. high_key_score / dark_score using the exact tone_curve.py logic
    high_key_score = float(
        smoothstep(0.55, 0.80, stats["r_high"])
        * smoothstep(0.58, 0.72, stats["p50"])
    )
    high_key_p50 = float(smoothstep(0.60, 0.80, stats["p50"])) * 0.8
    high_key_score = float(np.clip(max(high_key_score, high_key_p50), 0.0, 1.0))

    dark_score = float(
        smoothstep(0.45, 0.70, stats["r_low"])
        * (1.0 - smoothstep(0.32, 0.50, stats["p50"]))
    )
    dark_score = float(np.clip(dark_score, 0.0, 1.0))

    hks = high_key_score
    dks = dark_score

    # 4. Compute s_high / s_low
    s = BASE_S
    s_high = s * (1.0 - hks) + (s * S_SCALE_UI_HIGH) * hks
    s_high = s_high * (1.0 - dks) + (s * S_SCALE_DARK) * dks
    s_high = float(np.clip(s_high, S_CLIP_LO, S_CLIP_HI))

    s_low = s * (1.0 - hks) + (s * S_SCALE_UI_LOW) * hks
    s_low = s_low * (1.0 - dks) + (s * S_SCALE_DARK) * dks
    s_low = float(np.clip(s_low, S_CLIP_LO, S_CLIP_HI))

    # 5. Clip the four p_adj values to [0.25, 0.85]
    p_mode = float(np.clip(p_mode_raw, P_CLIP_LO, P_CLIP_HI))
    p50_direct = float(np.clip(p50, P_CLIP_LO, P_CLIP_HI))
    p_iqr = float(np.clip(p_iqr_raw, P_CLIP_LO, P_CLIP_HI))
    p_centroid = float(np.clip(p_centroid_raw, P_CLIP_LO, P_CLIP_HI))

    # 6. Curve kwargs
    curve_kwargs = {
        "s_low": s_low,
        "brightness_boost": float(0.4 * LUX_STRENGTH * dark_score),
        "brightness_start": 0.10,
        "brightness_end": 0.55,
        "high_protect": float(max(0.15 * LUX_STRENGTH, 0.15 * high_key_score)),
        "high_start": 0.55,
        "high_end": 1.0,
        "shadow_protect": float(0.25 * dark_score),
        "shadow_end": 0.3,
    }

    methods = [
        ("p_mode",     p_mode,     "p_mode.png",     "#d62728"),
        ("p50_direct", p50_direct, "p50_direct.png", "#2ca02c"),
        ("IQR center", p_iqr,      "p_iqr.png",      "#1f77b4"),
        ("Centroid",   p_centroid, "p_centroid.png", "#9467bd"),
    ]

    # 7. Apply curve & save outputs
    imwrite_safe(out_dir / "original.png", img)

    method_outputs = {}
    for name, p_adj, fname, _color in methods:
        out_img = apply_curve_y_only_bgr(
            img, p=float(p_adj), s=s_high,
            standard="bt709", curve_kwargs=curve_kwargs, lux=LUX,
        )
        imwrite_safe(out_dir / fname, out_img)
        method_outputs[name] = out_img

    # 8. Composite visualization (figsize=(20,10), DPI=150)
    fig = plt.figure(figsize=(20, 10), dpi=150)
    gs = GridSpec(
        2, 5,
        height_ratios=[1.0, 1.05],
        hspace=0.20, wspace=0.04,
        left=0.04, right=0.99, top=0.93, bottom=0.075,
    )

    top_panels = [
        ("Original", img, None),
        (f"p_mode\np_adj={p_mode:.3f}",       method_outputs["p_mode"],     "#d62728"),
        (f"p50_direct\np_adj={p50_direct:.3f}", method_outputs["p50_direct"], "#2ca02c"),
        (f"IQR center\np_adj={p_iqr:.3f}",    method_outputs["IQR center"], "#1f77b4"),
        (f"Centroid\np_adj={p_centroid:.3f}", method_outputs["Centroid"],   "#9467bd"),
    ]
    for col, (title, image, color) in enumerate(top_panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(image_to_rgb_float_for_plot(image))
        ax.set_title(
            title, fontsize=11,
            color=color if color is not None else "black",
            fontweight="bold" if color is not None else "normal",
        )
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
        f"(lux={LUX}, s_high={s_high:.3f}, s_low={s_low:.3f}, "
        f"hks={hks:.3f}, dks={dks:.3f})",
        fontsize=12,
    )
    ax_bot.grid(True, linestyle=":", alpha=0.45)
    ax_bot.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)

    fig.suptitle(
        f"p_adj Selection Methods Comparison  —  image: {img_path.name}",
        fontsize=14, y=0.985,
    )

    composite_path = out_dir / "comparison.png"
    fig.savefig(composite_path, dpi=150)
    plt.close(fig)

    return {
        "name": img_path.name,
        "stem": stem,
        "p_mode": p_mode,
        "p50": p50_direct,
        "p_iqr": p_iqr,
        "p_centroid": p_centroid,
        "high_key_score": hks,
        "dark_score": dks,
        "s_high": s_high,
        "s_low": s_low,
        "out_dir": out_dir,
        "composite": composite_path,
    }


def collect_images(folder: Path):
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def main():
    image_paths = collect_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images in {INPUT_DIR}")

    results = []
    for p in image_paths:
        print(f"  processing {p.name} ...", flush=True)
        r = process_image(p)
        if r is not None:
            results.append(r)

    if not results:
        return

    # ─── Summary table ─────────────────────────────────────────────────────────
    name_col_w = max(len("Image"), max(len(r["name"]) for r in results))
    header = (
        f"{'Image':<{name_col_w}} | "
        f"{'p_mode':<6} | {'p50':<6} | {'p_iqr':<6} | {'p_centroid':<10} | "
        f"{'hks':<5} | {'dark':<4}"
    )
    sep = (
        "-" * name_col_w + "-+-"
        + "-" * 6 + "-+-"
        + "-" * 6 + "-+-"
        + "-" * 6 + "-+-"
        + "-" * 10 + "-+-"
        + "-" * 5 + "-+-"
        + "-" * 4
    )
    print()
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['name']:<{name_col_w}} | "
            f"{r['p_mode']:<6.4f} | "
            f"{r['p50']:<6.4f} | "
            f"{r['p_iqr']:<6.4f} | "
            f"{r['p_centroid']:<10.4f} | "
            f"{r['high_key_score']:<5.3f} | "
            f"{r['dark_score']:<4.2f}"
        )

    print(f"\nAll outputs saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
