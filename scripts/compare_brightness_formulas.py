"""
Compare three brightness score formulas across all test images.

Formulas:
    - Current: brightness_raw = p50
    - A:       brightness_raw = 0.25 * p15 + 0.35 * p50 + 0.40 * p75
    - B:       skew = (p75 - p50) - (p50 - p15);
               brightness_raw = p50 + 0.3 * skew

A common dead-zone mapping is then applied to map brightness_raw -> [-1, +1].

Usage:
    cd /mnt/data/wcm/sunlight_enhance
    python scripts/compare_brightness_formulas.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make project root importable when run via `python scripts/...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.utils.image_io import collect_image_paths, imread_unicode  # noqa: E402
from common.utils.image_utils import compute_luma_float  # noqa: E402


INPUT_DIR = PROJECT_ROOT / "test" / "input"
OUTPUT_DIR = PROJECT_ROOT / "test" / "output7" / "tone_compare"
OUTPUT_PLOT = OUTPUT_DIR / "brightness_formulas.png"


# ---------------------------------------------------------------------------
# Brightness mapping
# ---------------------------------------------------------------------------
def map_brightness(brightness_raw: float) -> float:
    """Apply the shared dead-zone mapping to [-1, +1]."""
    if brightness_raw <= 0.15:
        return -1.0
    if brightness_raw < 0.30:
        return -1.0 + (brightness_raw - 0.15) / 0.15
    if brightness_raw <= 0.55:
        return 0.0
    if brightness_raw < 0.80:
        return (brightness_raw - 0.55) / 0.25
    return 1.0


def brightness_current(p50: float) -> tuple[float, float]:
    raw = p50
    return raw, map_brightness(raw)


def brightness_a(p15: float, p50: float, p75: float) -> tuple[float, float]:
    raw = 0.25 * p15 + 0.35 * p50 + 0.40 * p75
    return raw, map_brightness(raw)


def brightness_b(p15: float, p50: float, p75: float) -> tuple[float, float]:
    skew = (p75 - p50) - (p50 - p15)
    raw = p50 + 0.3 * skew
    return raw, map_brightness(raw)


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------
def evaluate_image(path: Path) -> dict | None:
    img = imread_unicode(str(path))
    if img is None:
        print(f"[WARN] Failed to read: {path}")
        return None

    Y = compute_luma_float(img)  # BGR -> luma in [0,1]
    flat = Y.reshape(-1).astype(np.float32)

    max_samples = 1_000_000
    if flat.size > max_samples:
        step = max(1, flat.size // max_samples)
        flat = flat[::step]

    p15, p50, p75 = np.percentile(flat, [15, 50, 75])
    p15, p50, p75 = float(p15), float(p50), float(p75)

    raw_cur, b_cur = brightness_current(p50)
    raw_a, b_a = brightness_a(p15, p50, p75)
    raw_b, b_b = brightness_b(p15, p50, p75)

    return {
        "name": path.name,
        "p15": p15,
        "p50": p50,
        "p75": p75,
        "raw_current": raw_cur,
        "raw_A": raw_a,
        "raw_B": raw_b,
        "b_current": b_cur,
        "b_A": b_a,
        "b_B": b_b,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(records: list[dict]) -> None:
    header = (
        f"{'filename':<14} {'p15':>6} {'p50':>6} {'p75':>6} "
        f"{'raw_cur':>8} {'raw_A':>8} {'raw_B':>8} "
        f"{'b_cur':>7} {'b_A':>7} {'b_B':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['name']:<14} "
            f"{r['p15']:>6.3f} {r['p50']:>6.3f} {r['p75']:>6.3f} "
            f"{r['raw_current']:>8.3f} {r['raw_A']:>8.3f} {r['raw_B']:>8.3f} "
            f"{r['b_current']:>+7.3f} {r['b_A']:>+7.3f} {r['b_B']:>+7.3f}"
        )


def plot_comparison(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r["name"] for r in records]
    p15 = np.array([r["p15"] for r in records])
    p50 = np.array([r["p50"] for r in records])
    p75 = np.array([r["p75"] for r in records])
    b_cur = np.array([r["b_current"] for r in records])
    b_a = np.array([r["b_A"] for r in records])
    b_b = np.array([r["b_B"] for r in records])

    x = np.arange(len(records))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(12, 0.45 * len(records)), 9), sharex=True
    )

    # ---- Top: brightness scores ----
    width = 0.27
    ax_top.bar(x - width, b_cur, width, label="Current (p50)",
               color="#888888", alpha=0.85)
    ax_top.bar(x, b_a, width, label="A (0.25·p15 + 0.35·p50 + 0.40·p75)",
               color="#1f77b4", alpha=0.9)
    ax_top.bar(x + width, b_b, width, label="B (p50 + 0.3·skew)",
               color="#d62728", alpha=0.9)

    ax_top.axhline(0.0, color="black", linewidth=0.6)
    ax_top.axhline(1.0, color="gray", linewidth=0.4, linestyle="--")
    ax_top.axhline(-1.0, color="gray", linewidth=0.4, linestyle="--")
    ax_top.set_ylim(-1.15, 1.15)
    ax_top.set_ylabel("brightness score")
    ax_top.set_title(
        "Brightness score comparison: Current vs Formula A vs Formula B "
        "(sorted by p50)"
    )
    ax_top.grid(axis="y", linestyle=":", alpha=0.5)
    ax_top.legend(loc="lower right", fontsize=9)

    # ---- Bottom: percentile values ----
    ax_bot.plot(x, p15, marker="o", linewidth=1.4, label="p15",
                color="#2ca02c")
    ax_bot.plot(x, p50, marker="s", linewidth=1.4, label="p50",
                color="#ff7f0e")
    ax_bot.plot(x, p75, marker="^", linewidth=1.4, label="p75",
                color="#9467bd")

    # Highlight dead-zone band [0.15, 0.30] and [0.55, 0.80]
    ax_bot.axhspan(0.15, 0.30, color="#cccccc", alpha=0.25,
                   label="ramp [0.15, 0.30]")
    ax_bot.axhspan(0.55, 0.80, color="#cccccc", alpha=0.25,
                   label="ramp [0.55, 0.80]")

    ax_bot.set_ylabel("luma percentile value")
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bot.legend(loc="upper left", fontsize=9, ncol=2)
    ax_bot.set_xlabel("image (sorted by p50)")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    paths = collect_image_paths(INPUT_DIR, recursive=False)
    if not paths:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return 1

    print(f"Scanning {len(paths)} images in {INPUT_DIR} ...")
    records: list[dict] = []
    for p in paths:
        rec = evaluate_image(p)
        if rec is not None:
            records.append(rec)

    if not records:
        print("[ERROR] No valid records produced.")
        return 1

    # Sort by p50 ascending for readability
    records.sort(key=lambda r: r["p50"])

    print_table(records)

    plot_comparison(records, OUTPUT_PLOT)
    print()
    print(f"Saved plot to: {OUTPUT_PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
