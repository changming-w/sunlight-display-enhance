# scripts/apply_lut_bank.py
# Apply parameterized 1D LUT bank to input images.
# Outputs:
#   1. Enhanced results per lux level: output_dir/{lux}lux/*.png
#   2. Composite visualization with LUT curves: output_dir/composite/*.png


import sys
import argparse
from pathlib import Path


import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from common.utils.image_io import imread_unicode, imwrite_unicode, collect_image_paths
from instrument.board_lut_v2 import (
    LUTBankBuilder,
    board_interpolate_lut,
    compute_adapted_params,
)
from instrument.config import get_tone_params_by_lux


# Matplotlib CJK font support
for font in ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]:
    try:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

def ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    """Ensure input image is BGR uint8 3-channel."""
    if img is None:
        raise ValueError("Input image is None")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.integer):
            maxv = np.iinfo(img.dtype).max
            img = np.clip(img.astype(np.float32) * 255.0 / float(maxv), 0, 255).astype(np.uint8)
        else:
            img = np.clip(img.astype(np.float32), 0.0, 1.0)
            img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] != 3:
            raise ValueError(f"Unsupported channel number: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return img


def compute_y_histogram(Y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute p25, p35, p50, p75, p95 from Y channel float [0,1]."""
    flat = Y.ravel()
    p25 = float(np.percentile(flat, 25))
    p35 = float(np.percentile(flat, 35))
    p50 = float(np.percentile(flat, 50))
    p75 = float(np.percentile(flat, 75))
    p95 = float(np.percentile(flat, 95))
    return p25, p35, p50, p75, p95


def safe_lux_name(lux: int | float) -> str:
    """Generate lux folder name."""
    if float(lux).is_integer():
        return f"{int(lux)}lux"
    return f"{lux:g}lux"

def apply_lut_to_color_image(img_bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply 1D LUT to YCrCb-Y channel of BGR image.
    
    Args:
        img_bgr: Input BGR image
        lut: 1D LUT array, 8-bit [0,255] or 10-bit [0,1023]
    
    Returns:
        Enhanced BGR image
    """
    img_bgr = ensure_bgr_u8(img_bgr)

    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = img_ycrcb[:, :, 0].astype(np.uint8)

    if lut.max() > 255:
        lut_8 = np.clip(
            lut.astype(np.float32) * 255.0 / 1023.0 + 0.5,
            0,
            255
        ).astype(np.uint8)
    else:
        lut_8 = np.clip(lut, 0, 255).astype(np.uint8)

    y_out = lut_8[y]
    img_ycrcb[:, :, 0] = y_out

    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

def create_composite_output(
    img_orig: np.ndarray,
    img_enh_list: list[np.ndarray],
    lut_list: list[np.ndarray],
    stats_list: list[dict],
    lux_list: list[int],
    Y: np.ndarray | None = None,
    fig_w: float = 16,
):
    """
    Create composite visualization:
        Row 1: Original + enhanced images at different lux levels
        Row 2: LUT curves overlay with optional Y histogram
    """
    from matplotlib.gridspec import GridSpec

    n_imgs = 1 + len(lux_list)

    h, w = img_orig.shape[:2]
    aspect = h / w
    target_h = 4.0
    img_w = target_h / aspect

    fig_h = target_h + 3.2 + 0.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        2,
        n_imgs,
        figure=fig,
        height_ratios=[target_h, 3.2],
        width_ratios=[img_w] * n_imgs,
        hspace=0.08,
        wspace=0.02,
        left=0.04,
        right=0.97,
        top=0.97,
        bottom=0.08,
    )

    # Row 1: Original + enhanced images
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    ax0.set_title("Original", fontsize=10, fontweight="bold")
    ax0.axis("off")

    for i, (img_enh, stats, lux) in enumerate(zip(img_enh_list, stats_list, lux_list)):
        ax = fig.add_subplot(gs[0, 1 + i])
        ax.imshow(cv2.cvtColor(img_enh, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"lux={lux}\n"
            f"p={stats['p_adj']:.3f}  s_high={stats['s_high']:.3f}\n"
            f"s_low={stats['s_low']:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    # Row 2: LUT curves
    ax_curve = fig.add_subplot(gs[1, :])

    x = np.arange(256)
    identity = x / 255.0

    colors = [
        "#e74c3c",
        "#2ecc71",
        "#3498db",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#34495e",
    ]

    ax_curve.plot(x, identity, "k--", alpha=0.3, label="Identity")

    for i, (lut, stats, lux) in enumerate(zip(lut_list, stats_list, lux_list)):
        if lut.max() > 255:
            lut_plot = lut.astype(np.float32) / 1023.0
        else:
            lut_plot = lut.astype(np.float32) / 255.0

        color = colors[i % len(colors)]

        label = (
            f"lux={lux}  "
            f"p={stats['p_adj']:.3f}  "
            f"s_h={stats['s_high']:.3f}  "
            f"s_l={stats['s_low']:.3f}"
        )

        ax_curve.plot(
            x,
            lut_plot,
            color=color,
            linewidth=1.5,
            label=label
        )

        ax_curve.axvline(
            stats["p_iqr"] * 255,
            color=color,
            linestyle=":",
            alpha=0.4
        )

    # Y histogram overlay
    if Y is not None:
        ax_hist = ax_curve.twinx()
        hist_vals, _, _ = ax_hist.hist(
            (Y * 255).ravel(),
            bins=256,
            range=(0, 256),
            alpha=0.18,
            color="#8e44ad",
            density=True,
            label="Y Histogram",
        )

        max_density = hist_vals.max() if hist_vals.max() > 0 else 1.0
        scale_factor = 0.3 / max_density
        ax_hist.set_ylim(0, max_density / scale_factor * 1.05)
        ax_hist.set_ylabel("Density (scaled)", fontsize=8, color="#8e44ad")
        ax_hist.tick_params(axis="y", labelcolor="#8e44ad", labelsize=7)
        ax_hist.legend(loc="upper right", fontsize=7, framealpha=0.7)

    ax_curve.set_xlabel("Input Y (0-255)", fontsize=9)
    ax_curve.set_ylabel("Output Y", fontsize=9)

    if stats_list:
        ax_curve.set_title(
            f"LUT Curves  |  p50={stats_list[0]['p50']:.3f}  "
            f"p_iqr={stats_list[0]['p_iqr']:.3f}",
            fontsize=10,
        )
    else:
        ax_curve.set_title("LUT Curves", fontsize=10)

    ax_curve.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_curve.set_xlim(0, 255)
    ax_curve.set_ylim(0, 1.05)
    ax_curve.grid(True, alpha=0.3)

    return fig


def render_matplotlib_figure_to_bgr(fig) -> np.ndarray:
    """Render matplotlib figure to BGR uint8 image."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3]
    result_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return result_bgr

def main():
    parser = argparse.ArgumentParser(
        description="Apply Parameterized 1D LUT Bank to images"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="test/input1",
        help="Input image folder."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_lut",
        help="Output root folder."
    )
    parser.add_argument(
        "--lux",
        type=int,
        nargs="+",
        default=[100, 5000, 25000, 45000, 80000],
        help="Lux levels."
    )

    parser.add_argument(
        "--n-p50",
        type=int,
        default=8,
        help="Number of p50 grid points for LUT bank."
    )
    parser.add_argument(
        "--n-iqr",
        type=int,
        default=8,
        help="Number of iqr grid points for LUT bank."
    )
    parser.add_argument(
        "--sparse-step",
        type=int,
        default=1,
        help="Sparse step for LUT bank."
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[8, 10],
        help="LUT bit depth."
    )
    parser.add_argument(
        "--p-method",
        type=str,
        default="iqr",
        choices=["iqr", "median", "fixed"],
        help="Pivot adaptation method."
    )

    parser.add_argument(
        "--save-composite",
        action="store_true",
        default=True,
        help="Save composite visualization image."
    )
    parser.add_argument(
        "--no-save-composite",
        dest="save_composite",
        action="store_false",
        help="Disable saving composite visualization image."
    )

    parser.add_argument(
        "--save-enhanced",
        action="store_true",
        default=True,
        help="Save enhanced images into output_dir/{lux}lux folders."
    )
    parser.add_argument(
        "--no-save-enhanced",
        dest="save_enhanced",
        action="store_false",
        help="Disable saving enhanced images."
    )

    parser.add_argument(
        "--output-ext",
        type=str,
        default=".png",
        help="Output image extension, e.g. .png or .jpg."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"[Error] input-dir is not a valid directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_ext = args.output_ext
    if not output_ext.startswith("."):
        output_ext = "." + output_ext

    composite_dir = output_dir / "composite"
    if args.save_composite:
        composite_dir.mkdir(parents=True, exist_ok=True)

    lux_output_dirs: dict[int, Path] = {}
    if args.save_enhanced:
        for lux in args.lux:
            lux_dir = output_dir / safe_lux_name(lux)
            lux_dir.mkdir(parents=True, exist_ok=True)
            lux_output_dirs[lux] = lux_dir

    # Build LUT Bank
    print("=" * 80)
    print("Apply Parameterized 1D LUT Bank")
    print("=" * 80)
    print("Building LUT Bank...")

    builder = LUTBankBuilder(
        n_p50=args.n_p50,
        n_iqr=args.n_iqr,
        sparse_step=args.sparse_step,
        entries=256,
        lut_bits=args.bits,
        p_method=args.p_method,
        smooth=True,
    )

    bank = builder.build()

    print(
        f"  Bank: {len(bank['lux_centers'])} lux × "
        f"{len(bank['p50_sparse'])} p50 × "
        f"{len(bank['iqr_sparse'])} iqr = "
        f"{bank['luts'].shape[0] * bank['luts'].shape[1] * bank['luts'].shape[2]} LUTs"
    )

    print(f"  Input dir        : {input_dir}")
    print(f"  Output dir       : {output_dir}")
    print(f"  Lux levels       : {args.lux}")
    print(f"  LUT bits         : {args.bits}")
    print(f"  p method         : {args.p_method}")
    print(f"  Save enhanced    : {args.save_enhanced}")
    print(f"  Save composite   : {args.save_composite}")
    print("=" * 80)

    # Collect images
    images = collect_image_paths(input_dir)
    images = sorted(images)

    print(f"Found {len(images)} images\n")

    if not images:
        print("[Warning] No images found.")
        return

    # Process each image
    for idx, img_path in enumerate(images, 1):
        img_bgr = imread_unicode(str(img_path))

        if img_bgr is None:
            print(f"[{idx}/{len(images)}] SKIP cannot read: {img_path.name}")
            continue

        try:
            img_bgr = ensure_bgr_u8(img_bgr)
        except Exception as e:
            print(f"[{idx}/{len(images)}] SKIP invalid image: {img_path.name}, error={e}")
            continue

        # Compute Y channel statistics
        Y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        p25, p35, p50, p75, p95 = compute_y_histogram(Y)
        p_iqr = (p25 + p75) / 2.0

        stem = img_path.stem
        out_name = f"{stem}{output_ext}"

        img_enh_list: list[np.ndarray] = []
        lut_list: list[np.ndarray] = []
        stats_list: list[dict] = []

        print(
            f"[{idx}/{len(images)}] {img_path.name:40s} "
            f"p25={p25:.3f}  p35={p35:.3f}  p50={p50:.3f}  p75={p75:.3f}  p95={p95:.3f}  p_iqr={p_iqr:.3f}"
        )

        for lux in args.lux:
            # Interpolate LUT for current lux
            lut = board_interpolate_lut(lux, p50, p_iqr, bank)

            # Apply LUT to get enhanced image
            img_enh = apply_lut_to_color_image(img_bgr, lut)

            # Get adapted parameters for annotation
            p_base, s_base = get_tone_params_by_lux(lux)
            p_adj, s_high, s_low, _ = compute_adapted_params(
                p50,
                p_iqr,
                p_base,
                s_base,
                p_method=args.p_method
            )

            stats = {
                "p_adj": float(p_adj),
                "s_high": float(s_high),
                "s_low": float(s_low),
                "p50": float(p50),
                "p_iqr": float(p_iqr),
            }

            img_enh_list.append(img_enh)
            lut_list.append(lut)
            stats_list.append(stats)

            # Save enhanced image for current lux
            if args.save_enhanced:
                lux_dir = lux_output_dirs[lux]
                enh_out_path = lux_dir / out_name
                imwrite_unicode(str(enh_out_path), img_enh)

                print(
                    f"    lux={lux:<8d} "
                    f"p={p_adj:.3f}  s_high={s_high:.3f}  s_low={s_low:.3f}  "
                    f"saved={enh_out_path}"
                )
            else:
                print(
                    f"    lux={lux:<8d} "
                    f"p={p_adj:.3f}  s_high={s_high:.3f}  s_low={s_low:.3f}"
                )

        # Save composite visualization
        if args.save_composite:
            fig = create_composite_output(
                img_orig=img_bgr,
                img_enh_list=img_enh_list,
                lut_list=lut_list,
                stats_list=stats_list,
                lux_list=args.lux,
                Y=Y,
            )

            result_bgr = render_matplotlib_figure_to_bgr(fig)
            plt.close(fig)

            comp_out_path = composite_dir / out_name
            imwrite_unicode(str(comp_out_path), result_bgr)

            print(f"    composite saved={comp_out_path}")

    print("\nDone.")
    print(f"Output root: {output_dir.resolve()}")

    if args.save_enhanced:
        print("Enhanced image folders:")
        for lux in args.lux:
            print(f"  {lux}: {lux_output_dirs[lux].resolve()}")

    if args.save_composite:
        print(f"Composite folder: {composite_dir.resolve()}")


if __name__ == "__main__":
    main()