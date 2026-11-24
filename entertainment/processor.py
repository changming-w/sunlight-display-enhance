# entertainment_processor.py
#
# 娱乐屏处理流水线：
#   Stage1: 对数曲线增强 (Y-only)
#   Stage2: Sobel 边缘增强 (无描边)
#   Stage3: 白平衡漂移 (与仪表屏共用)
#
# 提供与 processor.py 一致的对比图与边缘可视化输出：
#   - _compare_stage1/   Stage1 对数曲线对比图（娱乐屏定制）
#   - _compare_stage2/   Stage2 Sobel 对比图
#   - _compare_stage3/   Stage3 WB 对比图
#   - _edge_pipeline/    Stage2 ΔY 流水线可视化

from pathlib import Path

import numpy as np

from instrument.config import lux_to_name, get_wb_params_by_lux
from entertainment.config import ENTERTAINMENT_EDGE_PRESETS
from entertainment.tone_curve import (
    get_entertainment_params_by_lux,
    enhance_image_entertainment,
    entertainment_curve,
)
from common.algorithm.edge_enhance import (
    sobel_edge_enhance_image,
    prepare_edge_params_for_image,
    extract_y_from_image_float,
    _find_exact_key,
    _sanitize_edge_params,
)
from common.utils.image_io import imread_unicode, imwrite_unicode, collect_image_paths
from common.utils.image_utils import image_to_rgb_float_for_plot
from common.utils.metrics import compute_contrast_metrics
from common.algorithm.white_balance import WhiteBalanceShifter

from visualization.stage_compare import (
    save_stage2_comparison_plot,
    save_stage3_comparison_plot,
    save_edge_pipeline_visualization_plot,
)


# ============================================================
# Edge / WB helpers
# ============================================================

def _get_scaled_edge_params(lux, edge_scale=1.0):
    """获取娱乐屏 Sobel 边缘增强参数（纯锐化，无描边），并应用全局 amount scale。"""
    lux = float(lux)
    presets = ENTERTAINMENT_EDGE_PRESETS
    preset_lux = sorted(presets.keys())

    exact_key = _find_exact_key(lux, preset_lux)
    if exact_key is not None:
        edge_params = dict(presets[exact_key])
    elif lux <= preset_lux[0]:
        edge_params = dict(presets[preset_lux[0]])
    elif lux >= preset_lux[-1]:
        edge_params = dict(presets[preset_lux[-1]])
    else:
        # 线性插值
        for i in range(len(preset_lux) - 1):
            l0, l1 = preset_lux[i], preset_lux[i + 1]
            if l0 <= lux <= l1:
                t = (lux - l0) / (l1 - l0)
                p0, p1 = presets[l0], presets[l1]
                edge_params = {}
                for key in p0:
                    v0, v1 = float(p0[key]), float(p1[key])
                    edge_params[key] = v0 * (1 - t) + v1 * t
                break
        else:
            edge_params = dict(presets[preset_lux[-1]])

    edge_params["amount"] = float(edge_params["amount"]) * float(edge_scale)
    return _sanitize_edge_params(edge_params)


def _get_wb_shifter_from_cache(cache, wb_params, preserve_y):
    key = (
        tuple(wb_params["src_white_xy"]),
        tuple(wb_params["dst_white_xy"]),
        bool(preserve_y),
    )

    if key not in cache:
        cache[key] = WhiteBalanceShifter(
            src_white_xy=wb_params["src_white_xy"],
            dst_white_xy=wb_params["dst_white_xy"],
            preserve_y=preserve_y,
        )

    return cache[key]


# ============================================================
# Stage1 对比图（娱乐屏定制：使用对数曲线 entertainment_curve）
# ============================================================

def _save_stage1_entertainment_comparison_plot(
    original_img,
    records,
    save_path,
):
    """
    娱乐屏 Stage1 对比图。

    布局：
        第一行：Original + 不同 lux 的 Stage1 对数曲线结果
        第二行：Y 直方图 | 对比度指标 | 对数曲线 LUT
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if records is None or len(records) == 0:
        return

    n_levels = len(records)
    n_imgs = 1 + n_levels

    fig_w = max(12.0, 3.5 * n_imgs)
    fig_h = 6.8

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.35],
        hspace=0.32,
    )

    gs_top = outer[0].subgridspec(1, n_imgs, wspace=0.08)
    gs_bottom = outer[1].subgridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.25],
        wspace=0.25,
    )

    # ---- Top row: images ----
    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image_to_rgb_float_for_plot(original_img), interpolation="nearest")
    ax.set_title("Original", fontsize=10)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        k = rec.get("k", 0.0)
        a = rec.get("a", 0.0)
        img = rec["stage1_img"]

        ax = fig.add_subplot(gs_top[0, i + 1])
        ax.imshow(image_to_rgb_float_for_plot(img), interpolation="nearest")
        ax.set_title(
            f"Stage1 Log\n{lux_to_name(lux)}, k={k:.2f}, a={a:.3f}",
            fontsize=10,
        )
        ax.axis("off")

    # ---- Bottom-left: Y histogram of Original ----
    ax_hist = fig.add_subplot(gs_bottom[0, 0])
    Y_orig = extract_y_from_image_float(original_img, standard="bt709")
    ax_hist.hist(
        Y_orig.reshape(-1),
        bins=32,
        range=(0.0, 1.0),
        density=True,
        alpha=0.9,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )
    ax_hist.set_title("Y Histogram (Original)", fontsize=10)
    ax_hist.set_xlabel("Y")
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_axisbelow(True)
    ax_hist.grid(True, alpha=0.5, linewidth=0.5, color="#cccccc")

    # ---- Bottom-mid: contrast metric bars ----
    ax_metric = fig.add_subplot(gs_bottom[0, 1])
    orig_m = compute_contrast_metrics(original_img)
    eps = 1e-8
    lux_names = []
    std_ratios = []
    range_ratios = []
    local_ratios = []
    for rec in records:
        m = compute_contrast_metrics(rec["stage1_img"])
        lux_names.append(lux_to_name(rec["lux"]))
        std_ratios.append(m["std_255"] / max(orig_m["std_255"], eps))
        range_ratios.append(m["p95_p5_255"] / max(orig_m["p95_p5_255"], eps))
        local_ratios.append(m["local_std_mean_255"] / max(orig_m["local_std_mean_255"], eps))

    x_pos = np.arange(len(records), dtype=np.float32)
    width = 0.24
    ax_metric.bar(x_pos - width, std_ratios, width, label="Y std", color="#1f77b4", alpha=0.9)
    ax_metric.bar(x_pos, range_ratios, width, label="P95-P5", color="#ff7f0e", alpha=0.9)
    ax_metric.bar(x_pos + width, local_ratios, width, label="Local std", color="#2ca02c", alpha=0.9)
    ax_metric.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.65, label="Original = 1.0")
    all_values = std_ratios + range_ratios + local_ratios
    ymax = max(max(all_values), 1.0) * 1.25 if len(all_values) > 0 else 1.25
    for i in range(len(records)):
        for xx, vv in zip(
            [x_pos[i] - width, x_pos[i], x_pos[i] + width],
            [std_ratios[i], range_ratios[i], local_ratios[i]],
        ):
            ax_metric.text(
                xx,
                vv + 0.025 * ymax,
                f"{vv:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_metric.set_title("Stage1 Contrast Metrics / Original", fontsize=10)
    ax_metric.set_xticks(x_pos)
    ax_metric.set_xticklabels(lux_names)
    ax_metric.set_ylabel("Ratio")
    ax_metric.set_ylim(0.0, ymax)
    ax_metric.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_metric.legend(loc="upper left", fontsize=8)

    # ---- Bottom-right: Entertainment Log curves ----
    ax_curve = fig.add_subplot(gs_bottom[0, 2])
    x_8bit = np.arange(256, dtype=np.float32)
    x = x_8bit / 255.0
    ax_curve.plot(
        x_8bit,
        x_8bit,
        linestyle="--",
        linewidth=1.3,
        color="#1f77b4",
        label="identity",
    )
    colors = [
        "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2",
    ]
    for i, rec in enumerate(records):
        lux = rec["lux"]
        k = rec.get("k", 10.0)
        a = rec.get("a", 0.0)

        y = entertainment_curve(x, k=k, a=a)
        y_8bit = np.clip(np.round(y * 255.0), 0, 255)
        color = colors[i % len(colors)]
        ax_curve.plot(
            x_8bit,
            y_8bit,
            linewidth=1.7,
            color=color,
            label=f"{lux_to_name(lux)}",
        )

    ax_curve.set_title("Entertainment Log LUTs", fontsize=10)
    ax_curve.set_xlim(0, 255)
    ax_curve.set_ylim(0, 255)
    ax_curve.set_xlabel("Input")
    ax_curve.set_ylabel("Output")
    ax_curve.grid(True, linestyle="-", alpha=0.35)
    ax_curve.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ============================================================
# Pipeline
# ============================================================

def process_folder_entertainment(
    input_dir,
    output_dir,
    lux_list,
    recursive=False,
    overwrite=True,
    save_compare=True,
    save_edge=True,
    edge_scale=1.0,
    edge_content_adaptive=True,

    # Stage3 WB
    enable_wb=False,
    wb_preserve_y=True,
    wb_transfer="gamma",
    wb_gamma=2.2,
):
    """
    娱乐屏批处理入口。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    image_paths = collect_image_paths(input_dir, recursive=recursive)

    if len(image_paths) == 0:
        print(f"No image found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Curve type: entertainment (log curve)")
    print(f"Save Stage2 Sobel edge: {save_edge}")
    print(f"Stage2 content adaptive: {edge_content_adaptive}")
    print(f"Enable Stage3 WB drift: {enable_wb}")
    print(f"Edge amount scale: {edge_scale}")
    print(f"Lux list: {lux_list}")

    print()
    print("Stage1 entertainment curve parameters:")
    for lux in lux_list:
        k, a = get_entertainment_params_by_lux(lux)
        print(f"  {lux_to_name(lux)}: k = {k:.2f}, a = {a:.4f}")

    if save_edge:
        print()
        print("Stage2 Sobel edge base parameters:")
        for lux in lux_list:
            edge_params = _get_scaled_edge_params(lux, edge_scale=edge_scale)
            print(
                f"  {lux_to_name(lux)}: "
                f"amount = {edge_params['amount']:.4f}, "
                f"outline = {edge_params.get('outline_strength', 0.0):.4f}, "
                f"sigma = {edge_params['sigma']:.4f}"
            )

    if enable_wb:
        print()
        print("Stage3 white balance drift parameters:")
        print(f"  preserve_y : {wb_preserve_y}")
        print(f"  transfer   : {wb_transfer}")
        print(f"  gamma      : {wb_gamma}")
        for lux in lux_list:
            wb_params = get_wb_params_by_lux(lux)
            print(
                f"  {lux_to_name(lux)}: "
                f"{wb_params['name']}, "
                f"src={wb_params['src_white_xy']}, "
                f"dst={wb_params['dst_white_xy']}"
            )

    print()
    print("=" * 70)
    print("Start processing images (entertainment)")
    print("=" * 70)

    wb_shifter_cache = {}
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths, 1):
        rel_path = img_path.relative_to(input_dir)
        print()
        print(f"[Image {idx:04d}/{total:04d}] {img_path}")

        original_img = imread_unicode(img_path)
        if original_img is None:
            print(f"  [Skip] Cannot read: {img_path}")
            continue

        records = []

        for lux in lux_list:
            lux_name = lux_to_name(lux)

            # =================================================
            # Stage1：对数曲线增强
            # =================================================
            k, a = get_entertainment_params_by_lux(lux)

            stage1_path = output_dir / "stage1_tone" / lux_name / rel_path

            if stage1_path.exists() and not overwrite:
                print(f"  [Skip] Stage1 exists: {stage1_path}")
                stage1_img = imread_unicode(stage1_path)
                if stage1_img is None:
                    print(f"  [Warning] Failed to read existing Stage1: {stage1_path}")
                    continue
            else:
                try:
                    stage1_img = enhance_image_entertainment(
                        original_img,
                        k=k,
                        a=a,
                        lux=lux,
                    )
                except Exception as e:
                    print(f"  [Failed] Stage1 {lux_name}, error: {e}")
                    continue

                imwrite_unicode(stage1_path, stage1_img)
                print(
                    f"  [Save] Stage1 {lux_name} -> {stage1_path} "
                    f"(k={k:.2f}, a={a:.4f})"
                )

            # =================================================
            # Stage2：Sobel 边缘增强
            # =================================================
            stage2_img = stage1_img
            edge_params_base = _get_scaled_edge_params(lux, edge_scale=edge_scale)
            edge_params_eff = dict(edge_params_base)
            edge_stats = None

            if save_edge:
                try:
                    edge_params_eff, edge_stats = prepare_edge_params_for_image(
                        stage1_img,
                        edge_params_base,
                        standard="bt709",
                        content_adaptive=edge_content_adaptive,
                    )
                except Exception as e:
                    print(f"  [Warning] Stage2 param adapt {lux_name}, error: {e}")
                    edge_params_eff = dict(edge_params_base)
                    edge_stats = None

                stage2_path = output_dir / "stage2_sobel" / lux_name / rel_path

                if stage2_path.exists() and not overwrite:
                    print(f"  [Skip] Stage2 exists: {stage2_path}")
                    stage2_img = imread_unicode(stage2_path)
                    if stage2_img is None:
                        print(f"  [Warning] Failed to read existing Stage2: {stage2_path}")
                        stage2_img = stage1_img
                else:
                    try:
                        stage2_img = sobel_edge_enhance_image(
                            stage1_img,
                            amount=edge_params_eff["amount"],
                            sigma=edge_params_eff["sigma"],
                            outline_strength=edge_params_eff.get("outline_strength", 0.0),
                            edge_low=edge_params_eff["edge_low"],
                            edge_high=edge_params_eff["edge_high"],
                            edge_blur=edge_params_eff["edge_blur"],
                            luma_low=edge_params_eff["luma_low"],
                            luma_high=edge_params_eff["luma_high"],
                            norm_percentile=edge_params_eff["norm_percentile"],
                            high_luma_low=edge_params_eff["high_luma_low"],
                            high_luma_high=edge_params_eff["high_luma_high"],
                            high_luma_suppress=edge_params_eff["high_luma_suppress"],
                            standard="bt709",
                            content_adaptive=False,
                        )
                    except Exception as e:
                        print(f"  [Failed] Stage2 {lux_name}, error: {e}")
                        stage2_img = stage1_img
                    else:
                        imwrite_unicode(stage2_path, stage2_img)
                        print(
                            f"  [Save] Stage2 {lux_name} -> {stage2_path} "
                            f"(amount={edge_params_eff['amount']:.4f}, "
                            f"sigma={edge_params_eff['sigma']:.4f})"
                        )

            # =================================================
            # Stage3：白平衡漂移
            # =================================================
            stage3_img = stage2_img
            final_img = stage2_img
            wb_params = None

            if enable_wb:
                wb_params = get_wb_params_by_lux(lux)

                stage3_path = output_dir / "stage3_wb" / lux_name / rel_path

                if stage3_path.exists() and not overwrite:
                    print(f"  [Skip] Stage3 WB exists: {stage3_path}")
                    stage3_img = imread_unicode(stage3_path)
                    if stage3_img is None:
                        print(f"  [Warning] Failed to read existing Stage3 WB: {stage3_path}")
                        stage3_img = stage2_img
                    final_img = stage3_img
                else:
                    try:
                        wb_shifter = _get_wb_shifter_from_cache(
                            wb_shifter_cache,
                            wb_params,
                            preserve_y=wb_preserve_y,
                        )
                        stage3_img, wb_stats = wb_shifter.apply_bgr_image(
                            stage2_img,
                            transfer=wb_transfer,
                            gamma=wb_gamma,
                            return_stats=True,
                        )
                    except Exception as e:
                        print(f"  [Failed] Stage3 WB {lux_name}, error: {e}")
                        stage3_img = stage2_img
                        final_img = stage2_img
                    else:
                        final_img = stage3_img
                        imwrite_unicode(stage3_path, stage3_img)
                        print(
                            f"  [Save] Stage3 WB {lux_name} "
                            f"({wb_params['name']}) -> {stage3_path} "
                            f"(OOG_pixel={wb_stats['out_of_gamut_ratio'] * 100:.4f}%, "
                            f"OOG_channel={wb_stats['out_of_gamut_channel_ratio'] * 100:.4f}%, "
                            f"min={wb_stats['min_linear']:.4f}, "
                            f"max={wb_stats['max_linear']:.4f})"
                        )

            records.append(
                {
                    "lux": lux,

                    # 娱乐屏曲线参数
                    "k": float(k),
                    "a": float(a),

                    # 兼容字段（save_stage1/2/3 等可视化函数依赖）
                    "p": 0.0,
                    "s": 0.0,
                    "p_base": 0.0,
                    "s_base": 0.0,
                    "curve_kwargs": {},
                    "tone_stats": None,

                    "stage1_img": stage1_img,
                    "stage2_img": stage2_img,
                    "stage3_img": stage3_img,
                    "final_img": final_img,

                    "edge_params": edge_params_eff,
                    "edge_params_base": edge_params_base,
                    "edge_params_effective": edge_params_eff,
                    "edge_stats": edge_stats,

                    "wb_params": wb_params,
                }
            )

        # =====================================================
        # 对比图
        # =====================================================
        if save_compare and len(records) > 0:
            stage1_compare_path = (
                output_dir
                / "_compare_stage1"
                / rel_path.with_suffix(".stage1_compare.png")
            )
            stage2_compare_path = (
                output_dir
                / "_compare_stage2"
                / rel_path.with_suffix(".stage2_compare.png")
            )
            stage3_compare_path = (
                output_dir
                / "_compare_stage3"
                / rel_path.with_suffix(".stage3_compare.png")
            )
            edge_pipeline_path = (
                output_dir
                / "_edge_pipeline"
                / rel_path.with_suffix(".edge_pipeline.png")
            )

            if stage1_compare_path.exists() and not overwrite:
                print(f"  [Skip] Stage1 compare exists: {stage1_compare_path}")
            else:
                try:
                    _save_stage1_entertainment_comparison_plot(
                        original_img=original_img,
                        records=records,
                        save_path=stage1_compare_path,
                    )
                    print(f"  [Save] Stage1 compare -> {stage1_compare_path}")
                except Exception as e:
                    print(
                        f"  [Failed] Save Stage1 compare: "
                        f"{stage1_compare_path}, error: {e}"
                    )

            if save_edge:
                if stage2_compare_path.exists() and not overwrite:
                    print(f"  [Skip] Stage2 compare exists: {stage2_compare_path}")
                else:
                    try:
                        save_stage2_comparison_plot(
                            original_img=original_img,
                            records=records,
                            save_path=stage2_compare_path,
                        )
                        print(f"  [Save] Stage2 compare -> {stage2_compare_path}")
                    except Exception as e:
                        print(
                            f"  [Failed] Save Stage2 compare: "
                            f"{stage2_compare_path}, error: {e}"
                        )

            if enable_wb:
                if stage3_compare_path.exists() and not overwrite:
                    print(f"  [Skip] Stage3 compare exists: {stage3_compare_path}")
                else:
                    try:
                        save_stage3_comparison_plot(
                            original_img=original_img,
                            records=records,
                            save_path=stage3_compare_path,
                        )
                        print(f"  [Save] Stage3 compare -> {stage3_compare_path}")
                    except Exception as e:
                        print(
                            f"  [Failed] Save Stage3 compare: "
                            f"{stage3_compare_path}, error: {e}"
                        )

            if save_edge:
                if edge_pipeline_path.exists() and not overwrite:
                    print(f"  [Skip] Edge pipeline exists: {edge_pipeline_path}")
                else:
                    try:
                        save_edge_pipeline_visualization_plot(
                            records=records,
                            save_path=edge_pipeline_path,
                        )
                        print(f"  [Save] Edge pipeline -> {edge_pipeline_path}")
                    except Exception as e:
                        print(
                            f"  [Failed] Save Edge pipeline: "
                            f"{edge_pipeline_path}, error: {e}"
                        )
