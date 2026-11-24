"""
Simulate the perceived appearance of a display under various ambient
illuminance levels (lux).

Physical model
--------------
A display with peak luminance ``L_display_max`` (nits) viewed under ambient
illuminance ``lux`` reflects diffuse light off its front surface:

    L_reflected   = lux * reflectance / pi
    contrast_ratio = L_display_max / L_reflected
    perceived = image * (CR / (CR + 1)) + 1.0 / (CR + 1)

where ``image`` is the screen output normalised to [0, 1].  As ``lux``
increases, blacks are washed out and overall contrast collapses.

Usage
-----
    python scripts/simulate_ambient_viewing.py \
        --input ./test/input \
        --enhanced ./test/output19_tone_first/stage2_sobel/80000lux \
        --output ./test/simulate_viewing \
        --lux 5000 25000 45000 80000 \
        --display_nits 1000 \
        --reflectance 0.015
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Make project root importable when run via `python scripts/...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.utils.image_io import (  # noqa: E402
    collect_image_paths,
    imread_unicode,
    imwrite_unicode,
)


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------
def compute_contrast_ratio(lux: float, display_nits: float, reflectance: float) -> float:
    """Compute on-screen contrast ratio (full-on/full-off) under given ambient illuminance.

    Traditional definition: CR = (L_display_max + L_reflected) / L_reflected
    """
    if lux <= 0:
        return float("inf")
    l_reflected = lux * reflectance / math.pi
    if l_reflected <= 0:
        return float("inf")
    return (display_nits + l_reflected) / l_reflected


GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA


def apply_ambient_perception(img_float: np.ndarray, contrast_ratio: float) -> np.ndarray:
    """Apply the ambient-light wash-out model to a normalised [0, 1] image.

    Simplified model (no gamma encode/decode):
      perceived = image * (CR-1)/CR + 1/CR
    """
    if not math.isfinite(contrast_ratio):
        return np.clip(img_float, 0.0, 1.0)
    cr = contrast_ratio
    perceived = img_float * ((cr - 1.0) / cr) + 1.0 / cr
    return np.clip(perceived, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
TOP_BAR_HEIGHT = 60
SEPARATOR_WIDTH = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_centered_text(canvas: np.ndarray, text: str, x_left: int, x_right: int) -> None:
    """Draw black text centered in the horizontal range [x_left, x_right]."""
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    cx = (x_left + x_right) // 2
    org = (cx - tw // 2, TOP_BAR_HEIGHT // 2 + th // 2)
    cv2.putText(canvas, text, org, FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def build_comparison_image(
    raw_orig_u8: np.ndarray,
    perceived_orig_u8: np.ndarray,
    perceived_enh_u8: np.ndarray,
    lux: float,
    contrast_ratio: float,
) -> np.ndarray:
    """Stack raw original, perceived original, and perceived enhanced side-by-side."""
    h_o, w_o = raw_orig_u8.shape[:2]

    # Match heights for perceived orig
    h_po, w_po = perceived_orig_u8.shape[:2]
    if h_po != h_o:
        new_w = int(round(w_po * (h_o / h_po)))
        perceived_orig_u8 = cv2.resize(perceived_orig_u8, (new_w, h_o), interpolation=cv2.INTER_AREA)
        w_po = new_w

    # Match heights for perceived enhanced
    h_e, w_e = perceived_enh_u8.shape[:2]
    if h_e != h_o:
        new_w = int(round(w_e * (h_o / h_e)))
        perceived_enh_u8 = cv2.resize(perceived_enh_u8, (new_w, h_o), interpolation=cv2.INTER_AREA)
        w_e = new_w

    separator = np.full((h_o, SEPARATOR_WIDTH, 3), 255, dtype=np.uint8)
    bottom = np.hstack([raw_orig_u8, separator, perceived_orig_u8, separator, perceived_enh_u8])

    total_w = bottom.shape[1]
    top_bar = np.full((TOP_BAR_HEIGHT, total_w, 3), 255, dtype=np.uint8)

    cr_text = "inf" if not math.isfinite(contrast_ratio) else f"{contrast_ratio:.1f}"

    # Labels for three panels
    label_raw = "Original (no ambient)"
    label_orig = f"Original @ {int(lux)} lux (CR={cr_text}:1)"
    label_enh = f"Enhanced @ {int(lux)} lux (CR={cr_text}:1)"

    _put_centered_text(top_bar, label_raw, 0, w_o)
    x1 = w_o + SEPARATOR_WIDTH
    _put_centered_text(top_bar, label_orig, x1, x1 + w_po)
    x2 = x1 + w_po + SEPARATOR_WIDTH
    _put_centered_text(top_bar, label_enh, x2, x2 + w_e)

    # Draw a thin separator between top bar and image area.
    cv2.line(top_bar, (0, TOP_BAR_HEIGHT - 1), (total_w - 1, TOP_BAR_HEIGHT - 1),
             (200, 200, 200), 1)

    return np.vstack([top_bar, bottom])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def to_float01(img_bgr_u8: np.ndarray) -> np.ndarray:
    if img_bgr_u8.ndim == 2:
        img_bgr_u8 = cv2.cvtColor(img_bgr_u8, cv2.COLOR_GRAY2BGR)
    elif img_bgr_u8.shape[2] == 4:
        img_bgr_u8 = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGRA2BGR)
    return img_bgr_u8.astype(np.float32) / 255.0


def to_uint8(img_float: np.ndarray) -> np.ndarray:
    return np.clip(img_float * 255.0 + 0.5, 0, 255).astype(np.uint8)


def process_pair(
    orig_path: Path,
    enh_path: Path,
    output_root: Path,
    lux_levels: list[float],
    display_nits: float,
    reflectance: float,
) -> None:
    orig_bgr = imread_unicode(orig_path)
    enh_bgr = imread_unicode(enh_path)
    if orig_bgr is None:
        print(f"[WARN] Failed to read original: {orig_path}")
        return
    if enh_bgr is None:
        print(f"[WARN] Failed to read enhanced: {enh_path}")
        return

    orig_f = to_float01(orig_bgr)
    enh_f = to_float01(enh_bgr)
    raw_orig_u8 = to_uint8(orig_f)

    for lux in lux_levels:
        cr = compute_contrast_ratio(lux, display_nits, reflectance)
        orig_perc = apply_ambient_perception(orig_f, cr)
        enh_perc = apply_ambient_perception(enh_f, cr)

        comp = build_comparison_image(
            raw_orig_u8,
            to_uint8(orig_perc),
            to_uint8(enh_perc),
            lux,
            cr,
        )

        # Only create lux subfolder when multiple lux levels are processed
        if len(lux_levels) > 1:
            out_dir = output_root / f"{int(lux)}lux"
        else:
            out_dir = output_root
        out_path = out_dir / orig_path.name
        imwrite_unicode(out_path, comp)
        print(f"  saved: {out_path}")


def process_pair_composite(
    orig_path: Path,
    enh_paths: dict,
    output_root: Path,
    lux_levels: list[float],
    display_nits: float,
    reflectance: float,
) -> None:
    """Generate a single composite image: 4 rows (one per lux) x 3 cols (input@lux | original | enhanced@lux)."""
    orig_bgr = imread_unicode(orig_path)
    if orig_bgr is None:
        print(f"[WARN] Failed to read original: {orig_path}")
        return

    orig_f = to_float01(orig_bgr)
    raw_orig_u8 = to_uint8(orig_f)

    # Target cell width
    CELL_W = 400
    h_orig, w_orig = raw_orig_u8.shape[:2]
    cell_h = int(round(h_orig * (CELL_W / w_orig)))

    def resize_cell(img):
        return cv2.resize(img, (CELL_W, cell_h), interpolation=cv2.INTER_AREA)

    raw_cell = resize_cell(raw_orig_u8)
    rows = []

    for lux in lux_levels:
        cr = compute_contrast_ratio(lux, display_nits, reflectance)
        cr_text = "inf" if not math.isfinite(cr) else f"{cr:.1f}"

        # Left: input under ambient
        orig_perc = apply_ambient_perception(orig_f, cr)
        left_cell = resize_cell(to_uint8(orig_perc))

        # Right: enhanced under ambient
        enh_bgr = imread_unicode(enh_paths.get(lux))
        if enh_bgr is None:
            right_cell = np.full_like(left_cell, 128)
        else:
            enh_f = to_float01(enh_bgr)
            enh_perc = apply_ambient_perception(enh_f, cr)
            right_cell = resize_cell(to_uint8(enh_perc))

        # Separator
        sep = np.full((cell_h, 2, 3), 255, dtype=np.uint8)

        # Row: [input@lux | sep | original(no ambient) | sep | enhanced@lux]
        row_img = np.hstack([left_cell, sep, raw_cell, sep, right_cell])

        # Row label bar
        row_w = row_img.shape[1]
        label_h = 40
        label_bar = np.full((label_h, row_w, 3), 255, dtype=np.uint8)

        l1 = f"Input @ {int(lux)}lux (CR={cr_text}:1)"
        l2 = "Original (no ambient)"
        l3 = f"Enhanced @ {int(lux)}lux (CR={cr_text}:1)"

        _put_label(label_bar, l1, 0, CELL_W, label_h)
        _put_label(label_bar, l2, CELL_W + 2, CELL_W + 2 + CELL_W, label_h)
        _put_label(label_bar, l3, 2 * (CELL_W + 2), 2 * (CELL_W + 2) + CELL_W, label_h)

        rows.append(np.vstack([label_bar, row_img]))

    # Row separator
    row_sep = np.full((4, rows[0].shape[1], 3), 220, dtype=np.uint8)
    parts = []
    for i, row in enumerate(rows):
        if i > 0:
            parts.append(row_sep)
        parts.append(row)

    composite = np.vstack(parts)
    out_path = output_root / orig_path.name
    imwrite_unicode(out_path, composite)
    print(f"  saved: {out_path}")


def _put_label(bar, text, x_left, x_right, bar_h):
    font_scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    cx = (x_left + x_right) // 2
    org = (cx - tw // 2, bar_h // 2 + th // 2)
    cv2.putText(bar, text, org, FONT, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate the perceived display under different ambient lux levels."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Folder of original (un-enhanced) screen images.")
    parser.add_argument("--enhanced", type=Path, default=None,
                        help="Folder of enhanced screen images "
                             "(same filenames as --input). Used in pair mode.")
    parser.add_argument("--enhanced_dir", type=Path, default=None,
                        help="Directory containing lux subfolders (e.g. 5000lux/, 80000lux/). "
                             "Used in composite mode.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output root folder.")
    parser.add_argument("--lux", nargs="+", type=float,
                        default=[5000, 25000, 45000, 80000],
                        help="Ambient illuminance levels in lux.")
    parser.add_argument("--display_nits", type=float, default=1000.0,
                        help="Display peak luminance in nits.")
    parser.add_argument("--reflectance", type=float, default=0.015,
                        help="Diffuse reflectance of the display (e.g. 0.015 with AR).")
    parser.add_argument("--composite", action="store_true",
                        help="Generate composite image (4 rows x 3 cols) per image. "
                             "Requires --enhanced_dir.")
    args = parser.parse_args()

    input_dir: Path = args.input
    output_root: Path = args.output
    lux_levels: list[float] = list(args.lux)

    if not input_dir.is_dir():
        raise SystemExit(f"--input is not a directory: {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    print("Ambient viewing simulation")
    print(f"  display_nits = {args.display_nits}")
    print(f"  reflectance  = {args.reflectance}")
    print(f"  lux levels   = {lux_levels}")
    for lux in lux_levels:
        cr = compute_contrast_ratio(lux, args.display_nits, args.reflectance)
        cr_str = "inf" if not math.isfinite(cr) else f"{cr:.2f}"
        print(f"  -> {int(lux)} lux: CR = {cr_str}:1")

    orig_paths = collect_image_paths(input_dir)
    if not orig_paths:
        raise SystemExit(f"No images found in {input_dir}")

    if args.composite:
        # Composite mode: one big image per input
        if args.enhanced_dir is None:
            raise SystemExit("--composite requires --enhanced_dir")
        if not args.enhanced_dir.is_dir():
            raise SystemExit(f"--enhanced_dir is not a directory: {args.enhanced_dir}")

        processed = 0
        for orig_path in orig_paths:
            # Build enh_paths dict: {lux: path_to_enhanced_image}
            enh_paths = {}
            for lux in lux_levels:
                lux_dir = args.enhanced_dir / f"{int(lux)}lux"
                enh_file = lux_dir / orig_path.name
                if enh_file.exists():
                    enh_paths[lux] = enh_file
            print(f"Processing {orig_path.name}")
            process_pair_composite(
                orig_path,
                enh_paths,
                output_root,
                lux_levels,
                args.display_nits,
                args.reflectance,
            )
            processed += 1
        print(f"Done. Processed {processed} composite image(s).")
    elif args.enhanced is not None:
        # Pair mode: compare original vs enhanced under ambient
        enhanced_dir = args.enhanced
        if not enhanced_dir.is_dir():
            raise SystemExit(f"--enhanced is not a directory: {enhanced_dir}")

        enhanced_names = {p.name: p for p in collect_image_paths(enhanced_dir)}

        processed = 0
        for orig_path in orig_paths:
            enh_path = enhanced_names.get(orig_path.name)
            if enh_path is None:
                print(f"[SKIP] No enhanced counterpart for {orig_path.name}")
                continue
            print(f"Processing {orig_path.name}")
            process_pair(
                orig_path,
                enh_path,
                output_root,
                lux_levels,
                args.display_nits,
                args.reflectance,
            )
            processed += 1

        print(f"Done. Processed {processed} image(s).")
    else:
        # Simple / matched mode:
        # - Always generate input images under each lux
        # - If --enhanced_dir is provided, also generate enhanced images under matched lux
        has_enh_dir = (args.enhanced_dir is not None and args.enhanced_dir.is_dir())

        processed = 0
        for orig_path in orig_paths:
            orig_bgr = imread_unicode(orig_path)
            if orig_bgr is None:
                print(f"[WARN] Failed to read: {orig_path}")
                continue
            img_f = to_float01(orig_bgr)
            raw_orig_u8 = to_uint8(img_f)
            print(f"Processing {orig_path.name}")

            for lux in lux_levels:
                cr = compute_contrast_ratio(lux, args.display_nits, args.reflectance)

                # Input under ambient
                perceived_input = apply_ambient_perception(img_f, cr)
                in_dir = output_root / f"input_{int(lux)}lux"
                in_path = in_dir / orig_path.name
                imwrite_unicode(in_path, to_uint8(perceived_input))
                print(f"  saved: {in_path}")

                # Enhanced under matched ambient (if available)
                if has_enh_dir:
                    enh_file = args.enhanced_dir / f"{int(lux)}lux" / orig_path.name
                    if enh_file.exists():
                        enh_bgr = imread_unicode(enh_file)
                        if enh_bgr is not None:
                            enh_f = to_float01(enh_bgr)
                            perceived_enh = apply_ambient_perception(enh_f, cr)
                            enh_out_dir = output_root / f"enhanced_{int(lux)}lux"
                            enh_out_path = enh_out_dir / orig_path.name
                            imwrite_unicode(enh_out_path, to_uint8(perceived_enh))
                            print(f"  saved: {enh_out_path}")

                            # Comparison image: original(dark) | original@lux | enhanced@lux
                            comp = build_comparison_image(
                                raw_orig_u8,
                                to_uint8(perceived_input),
                                to_uint8(perceived_enh),
                                lux,
                                cr,
                            )
                            comp_dir = output_root / f"comparison_{int(lux)}lux"
                            comp_path = comp_dir / orig_path.name
                            imwrite_unicode(comp_path, comp)
                            print(f"  saved: {comp_path}")

            processed += 1
        print(f"Done. Processed {processed} image(s).")


if __name__ == "__main__":
    main()
