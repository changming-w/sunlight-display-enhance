# visualization.py

from pathlib import Path

import numpy as np

from instrument.config import lux_to_name, get_tone_params_by_lux
from common.utils.image_utils import image_to_rgb_float_for_plot
from common.utils.metrics import compute_contrast_metrics
from common.algorithm.curve_math import sunlight_curve
from instrument.tone_curve import enhance_image
from instrument.tone_curve_high_weaken import sunlight_curve_high_weaken

from common.algorithm.edge_enhance import (
    compute_sobel_debug_image,
    extract_y_from_image_float,
    sobel_edge_enhance_image,
    get_edge_params_by_lux,
)


def _plot_stage1_metric_bars(ax, original_img, records):
    """
    Stage1 对比度倍率柱状图。
    """
    orig_m = compute_contrast_metrics(original_img)
    eps = 1e-8

    lux_names = []
    std_ratios = []
    range_ratios = []
    local_ratios = []

    for rec in records:
        lux = rec["lux"]
        img = rec["stage1_img"]

        m = compute_contrast_metrics(img)

        lux_names.append(lux_to_name(lux))
        std_ratios.append(m["std_255"] / max(orig_m["std_255"], eps))
        range_ratios.append(m["p95_p5_255"] / max(orig_m["p95_p5_255"], eps))
        local_ratios.append(m["local_std_mean_255"] / max(orig_m["local_std_mean_255"], eps))

    x = np.arange(len(records), dtype=np.float32)
    width = 0.24

    ax.bar(x - width, std_ratios, width, label="Y std", color="#1f77b4", alpha=0.9)
    ax.bar(x, range_ratios, width, label="P95-P5", color="#ff7f0e", alpha=0.9)
    ax.bar(x + width, local_ratios, width, label="Local std", color="#2ca02c", alpha=0.9)

    ax.axhline(
        1.0,
        color="k",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="Original = 1.0",
    )

    all_values = std_ratios + range_ratios + local_ratios
    ymax = max(max(all_values), 1.0) * 1.25 if len(all_values) > 0 else 1.25

    for i in range(len(records)):
        for xx, vv in zip(
            [x[i] - width, x[i], x[i] + width],
            [std_ratios[i], range_ratios[i], local_ratios[i]],
        ):
            ax.text(
                xx,
                vv + 0.025 * ymax,
                f"{vv:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Stage1 Contrast Metrics / Original", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(lux_names)
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, ymax)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8)


def save_stage1_comparison_plot(
    original_img,
    records,
    save_path,
    mode="ycbcr",
):
    """
    Stage1 对比图：
        第一行：Original + 不同 lux 的 Stage1 Tone 结果
        第二行：对比度指标 + Tone LUT 曲线
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

    gs_top = outer[0].subgridspec(
        1,
        n_imgs,
        wspace=0.08,
    )

    gs_bottom = outer[1].subgridspec(
        1,
        2,
        width_ratios=[1.2, 1.0],
        wspace=0.25,
    )

    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image_to_rgb_float_for_plot(original_img), interpolation="nearest")
    ax.set_title("Original", fontsize=10)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        p = rec.get("p", rec.get("p_base", 0.0))
        s = rec.get("s", rec.get("s_base", 0.0))
        s_low = rec.get("curve_kwargs", {}).get("s_low", s)
        img = rec["stage1_img"]

        ax = fig.add_subplot(gs_top[0, i + 1])
        ax.imshow(image_to_rgb_float_for_plot(img), interpolation="nearest")
        ax.set_title(
            f"Stage1 Tone\n{lux_to_name(lux)}, p={p:.2f}, s_h={s:.2f}, s_l={s_low:.2f}",
            fontsize=10,
        )
        ax.axis("off")

    # --- Combined subplot: Histogram + Tone Curves ---
    ax_hist = fig.add_subplot(gs_bottom[0, 0])

    if mode == "hsv":
        # HSV 方案：显示 V 通道直方图（参数基于 V 统计适应）
        import cv2
        img_float = original_img.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
        hist_data = hsv[:, :, 2].reshape(-1)
        hist_label = "V Histogram"
        hist_xlabel = "V (normalized)"
        hist_title = "V Histogram + Tone Curves"
    else:
        # YCbCr 方案：显示 Y 通道直方图
        hist_data = extract_y_from_image_float(original_img, standard="bt709").reshape(-1)
        hist_label = "Y Histogram"
        hist_xlabel = "Y (normalized)"
        hist_title = "Y Histogram + Tone Curves"

    ax_hist.hist(
        hist_data,
        bins=64,
        range=(0.0, 1.0),
        density=True,
        alpha=0.5,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
        label=hist_label,
    )

    ax_hist.set_xlabel(hist_xlabel)
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_axisbelow(True)
    ax_hist.grid(True, alpha=0.35, linewidth=0.5, color="#cccccc")

    # Secondary axis for tone curves
    ax_curve = ax_hist.twinx()

    x = np.linspace(0.0, 1.0, 256)

    ax_curve.plot(
        x,
        x,
        linestyle="--",
        linewidth=1.3,
        color="gray",
        alpha=0.7,
        label="identity",
    )

    colors = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    for i, rec in enumerate(records):
        lux = rec["lux"]
        p = rec.get("p", rec.get("p_base", 0.0))
        s = rec.get("s", rec.get("s_base", 0.0))
        curve_kwargs = rec.get("curve_kwargs", {})

        # 根据是否启用 high_weaken 选择曲线函数
        hw = rec.get("high_weaken")
        if hw is not None:
            hw_start = rec.get("high_weaken_start", 200 / 255.0)
            hw_end = rec.get("high_weaken_end", 240 / 255.0)
            y = sunlight_curve_high_weaken(
                x, p=p, s=s,
                high_weaken=hw,
                high_weaken_start=hw_start,
                high_weaken_end=hw_end,
                **curve_kwargs,
            )
        else:
            y = sunlight_curve(x, p=p, s=s, **curve_kwargs)

        color = colors[i % len(colors)]

        ax_curve.plot(
            x,
            y,
            linewidth=1.7,
            color=color,
            label=f"{lux_to_name(lux)}",
        )

        ax_curve.scatter(
            [p],
            [p],
            s=18,
            color=color,
            zorder=5,
        )

    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.set_ylabel("Output")

    # Combine legends from both axes
    handles_hist, labels_hist = ax_hist.get_legend_handles_labels()
    handles_curve, labels_curve = ax_curve.get_legend_handles_labels()
    ax_hist.legend(
        handles_hist + handles_curve,
        labels_hist + labels_curve,
        loc="upper left",
        fontsize=8,
    )

    ax_hist.set_title(hist_title, fontsize=10)

    # --- Metric bars subplot (unchanged) ---
    ax_metric = fig.add_subplot(gs_bottom[0, 1])
    _plot_stage1_metric_bars(ax_metric, original_img, records)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _compute_delta_y_for_record(rec):
    """
    计算 Stage2 相对 Stage1 的 |Delta Y|。
    """
    stage1_img = rec["stage1_img"]
    stage2_img = rec["stage2_img"]

    Y1 = extract_y_from_image_float(stage1_img, standard="bt709")
    Y2 = extract_y_from_image_float(stage2_img, standard="bt709")

    delta_y = np.abs(Y2 - Y1).astype(np.float32)
    return delta_y


def save_stage2_comparison_plot(
    original_img,
    records,
    save_path,
):
    """
    第二阶段对比图。
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if records is None or len(records) == 0:
        return

    n_levels = len(records)
    n_cols = n_levels + 1

    delta_list = [
        _compute_delta_y_for_record(rec)
        for rec in records
    ]

    all_delta = np.concatenate(
        [d.reshape(-1) for d in delta_list],
        axis=0,
    )

    delta_vmax = float(np.percentile(all_delta, 99))
    delta_vmax = max(delta_vmax, 1.0 / 255.0)

    fig_w = max(15.0, 4.0 * n_cols)
    fig_h = 8.8

    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(fig_w, fig_h),
        dpi=150,
        squeeze=False,
    )

    ax = axes[0, 0]
    ax.imshow(image_to_rgb_float_for_plot(original_img), interpolation="nearest")
    ax.set_title("Original", fontsize=11)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        stage1_img = rec["stage1_img"]

        ax = axes[0, i + 1]
        ax.imshow(image_to_rgb_float_for_plot(stage1_img), interpolation="nearest")
        ax.set_title(f"Stage1 Tone {lux_to_name(lux)}", fontsize=11)
        ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(image_to_rgb_float_for_plot(original_img), interpolation="nearest")
    ax.set_title("Original", fontsize=11)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        stage2_img = rec["stage2_img"]

        ax = axes[1, i + 1]
        ax.imshow(image_to_rgb_float_for_plot(stage2_img), interpolation="nearest")
        ax.set_title(f"Stage2 Sobel {lux_to_name(lux)}", fontsize=11)
        ax.axis("off")

    ax = axes[2, 0]
    ax.axis("off")
    ax.set_title("", fontsize=11)

    for i, rec in enumerate(records):
        lux = rec["lux"]
        delta_y = delta_list[i]

        ax = axes[2, i + 1]
        ax.imshow(
            delta_y,
            cmap="gray",
            vmin=0.0,
            vmax=delta_vmax,
            interpolation="nearest",
        )
        ax.set_title(f"|Delta Y| {lux_to_name(lux)}", fontsize=11)
        ax.axis("off")

    plt.tight_layout(pad=0.35)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_stage3_comparison_plot(
    original_img,
    records,
    save_path,
):
    """
    第三阶段白平衡漂移对比图。
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if records is None or len(records) == 0:
        return

    n_levels = len(records)
    n_cols = n_levels + 1

    fig_w = max(15.0, 4.0 * n_cols)
    fig_h = 6.2

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_w, fig_h),
        dpi=150,
        squeeze=False,
    )

    ax = axes[0, 0]
    ax.imshow(image_to_rgb_float_for_plot(original_img), interpolation="nearest")
    ax.set_title("Original", fontsize=11)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        stage2_img = rec["stage2_img"]

        ax = axes[0, i + 1]
        ax.imshow(image_to_rgb_float_for_plot(stage2_img), interpolation="nearest")
        ax.set_title(f"Stage2 Sobel {lux_to_name(lux)}", fontsize=11)
        ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(image_to_rgb_float_for_plot(original_img), interpolation="nearest")
    ax.set_title("Original", fontsize=11)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        stage3_img = rec.get("stage3_img", rec.get("final_img", rec["stage2_img"]))
        wb_params = rec.get("wb_params", None)

        if wb_params is not None:
            wb_name = wb_params.get("name", "")
            dst_xy = wb_params.get("dst_white_xy", None)

            if wb_name != "":
                title = f"Stage3 WB {lux_to_name(lux)}\n{wb_name}"
            elif dst_xy is not None:
                title = f"Stage3 WB {lux_to_name(lux)}\nxy=({dst_xy[0]:.4f},{dst_xy[1]:.4f})"
            else:
                title = f"Stage3 WB {lux_to_name(lux)}"
        else:
            title = f"Stage3 WB {lux_to_name(lux)}"

        ax = axes[1, i + 1]
        ax.imshow(image_to_rgb_float_for_plot(stage3_img), interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout(pad=0.35)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_edge_pipeline_visualization_plot(
    records,
    save_path,
):
    """
    Stage1 -> Stage2 deltaY 多 lux 可视化图。

    每一行代表一种光照强度 lux：

        Stage1 Tone Output | DeltaY(Stage2 - Stage1) | Stage2 Sobel Output

    说明：
        1. 第一列：Stage1 Tone 输出，也就是 Stage2 Sobel 的输入；
        2. 第二列：deltaY = Y(Stage2) - Y(Stage1)，是有符号亮度差；
        3. 第三列：Stage2 Sobel 输出；
        4. 不显示 Stage3 WB，避免和白平衡漂移混淆。
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if records is None or len(records) == 0:
        return

    # 保持 processor.py 中 records 的顺序，
    # 也就是保持用户输入 lux_list 的顺序。
    records_to_plot = list(records)

    n_rows = len(records_to_plot)
    n_cols = 3

    # ========================================================
    # 预先计算所有 lux 的 deltaY，用统一显示范围
    # ========================================================
    delta_list = []

    for rec in records_to_plot:
        stage1_img = rec["stage1_img"]
        stage2_img = rec["stage2_img"]

        Y1 = extract_y_from_image_float(stage1_img, standard="bt709")
        Y2 = extract_y_from_image_float(stage2_img, standard="bt709")

        # 有符号 deltaY：
        #   > 0 表示 Stage2 比 Stage1 更亮
        #   < 0 表示 Stage2 比 Stage1 更暗
        delta_y = (Y2 - Y1).astype(np.float32)

        delta_list.append(delta_y)

    all_delta = np.concatenate(
        [d.reshape(-1) for d in delta_list],
        axis=0,
    )

    # 使用绝对值 99 分位作为显示范围，避免极端点影响可视化
    delta_abs_vmax = float(np.percentile(np.abs(all_delta), 99.0))

    # 防止全 0 或极小 delta 导致显示异常
    delta_abs_vmax = max(delta_abs_vmax, 1.0 / 255.0)

    # ========================================================
    # 创建画布
    # ========================================================
    fig_w = 15.0
    fig_h = max(4.0, 3.6 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        dpi=180,
        squeeze=False,
    )

    im_delta_for_colorbar = None

    for row_idx, rec in enumerate(records_to_plot):
        lux = rec["lux"]
        lux_name = lux_to_name(lux)

        stage1_img = rec["stage1_img"]
        stage2_img = rec["stage2_img"]

        delta_y = delta_list[row_idx]

        # ====================================================
        # 1. 第一列：Stage1 Tone 输出
        # ====================================================
        stage1_vis = image_to_rgb_float_for_plot(stage1_img)

        ax = axes[row_idx, 0]
        ax.imshow(stage1_vis, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"{lux_name}\nStage1 Tone Output",
            fontsize=10,
        )

        # ====================================================
        # 2. 第二列：DeltaY = Stage2 - Stage1
        # ====================================================
        ax = axes[row_idx, 1]

        im_delta = ax.imshow(
            delta_y,
            cmap="seismic",
            vmin=-delta_abs_vmax,
            vmax=delta_abs_vmax,
            interpolation="nearest",
        )

        im_delta_for_colorbar = im_delta

        ax.axis("off")

        mean_delta = float(np.mean(delta_y))
        mean_abs_delta = float(np.mean(np.abs(delta_y)))
        p95_abs_delta = float(np.percentile(np.abs(delta_y).reshape(-1), 95.0))

        ax.set_title(
            f"{lux_name}\nΔY = Y2 - Y1\n"
            f"mean={mean_delta * 255.0:+.2f}, "
            f"mean|Δ|={mean_abs_delta * 255.0:.2f}, "
            f"p95|Δ|={p95_abs_delta * 255.0:.2f}",
            fontsize=10,
        )

        # ====================================================
        # 3. 第三列：Stage2 Sobel 输出
        # ====================================================
        stage2_vis = image_to_rgb_float_for_plot(stage2_img)

        ax = axes[row_idx, 2]
        ax.imshow(stage2_vis, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"{lux_name}\nStage2 Sobel Output",
            fontsize=10,
        )

    # ========================================================
    # 添加共享 colorbar
    # ========================================================
    if im_delta_for_colorbar is not None:
        cbar = fig.colorbar(
            im_delta_for_colorbar,
            ax=axes[:, 1],
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("ΔY = Y(Stage2) - Y(Stage1)", fontsize=9)

    fig.suptitle(
        "Stage2 Sobel Enhancement DeltaY Visualization\n"
        "Red: Stage2 brighter than Stage1, Blue: Stage2 darker than Stage1",
        fontsize=12,
        y=0.995,
    )

    plt.tight_layout(pad=0.35, rect=[0.0, 0.0, 1.0, 0.965])
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _process_for_adaptive_compare(original_img, lux, mode, content_adaptive):
    """
    Helper：以指定 content_adaptive 设置，跑 Stage1 + Stage2，
    返回 (stage2_img, p_eff, s_eff, curve_kwargs)。
    """
    p_base, s_base = get_tone_params_by_lux(lux)

    stage1_result = enhance_image(
        original_img,
        p=p_base,
        s=s_base,
        mode=mode,
        content_adaptive=content_adaptive,
        return_info=True,
        lux=lux,
    )

    if isinstance(stage1_result, tuple):
        stage1_img, info = stage1_result
        p_eff = float(info.get("p", p_base))
        s_eff = float(info.get("s", s_base))
        curve_kwargs = dict(info.get("curve_kwargs", {}) or {})
    else:
        stage1_img = stage1_result
        p_eff = float(p_base)
        s_eff = float(s_base)
        curve_kwargs = {}

    edge_params = get_edge_params_by_lux(lux)

    stage2_img = sobel_edge_enhance_image(
        stage1_img,
        content_adaptive=content_adaptive,
        **edge_params,
    )

    return stage2_img, p_eff, s_eff, curve_kwargs


def save_adaptive_comparison_plot(
    original_img,
    save_path,
    mode="ycbcr",
):
    """
    自适应 vs 非自适应 Stage2 对比图（3 行 3 列）。

    布局：
        Col 0: Original
        Col 1: 45000lux Non-Adaptive
        Col 2: 45000lux Adaptive

        Row 0: Image (Original / Stage2 non-adapt / Stage2 adapt)
        Row 1: Tone curve (y=x ref / non-adapt curve / adapt curve)
        Row 2: Y histogram
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lux = 45000

    columns = []

    # Col 0: Original (no Stage2 processing, no tone curve)
    columns.append(
        {
            "kind": "original",
            "image": original_img,
            "title": "Original",
        }
    )

    # Col 1: 45000lux Non-Adaptive
    stage2_no_adapt, p_no, s_no, ck_no = _process_for_adaptive_compare(
        original_img, lux, mode, content_adaptive=False,
    )
    columns.append(
        {
            "kind": "processed",
            "lux": lux,
            "adaptive": False,
            "stage2_img": stage2_no_adapt,
            "p": p_no,
            "s": s_no,
            "curve_kwargs": ck_no,
            "title": f"{lux_to_name(lux)} Non-Adaptive",
            "curve_color": "#d62728",
        }
    )

    # Col 2: 45000lux Adaptive
    stage2_adapt, p_ad, s_ad, ck_ad = _process_for_adaptive_compare(
        original_img, lux, mode, content_adaptive=True,
    )
    columns.append(
        {
            "kind": "processed",
            "lux": lux,
            "adaptive": True,
            "stage2_img": stage2_adapt,
            "p": p_ad,
            "s": s_ad,
            "curve_kwargs": ck_ad,
            "title": f"{lux_to_name(lux)} Adaptive",
            "curve_color": "#2ca02c",
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), dpi=150)

    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    for col_idx, col in enumerate(columns):
        is_original = col["kind"] == "original"

        # Row 0: image
        ax = axes[0, col_idx]
        if is_original:
            img_to_show = col["image"]
        else:
            img_to_show = col["stage2_img"]
        ax.imshow(
            image_to_rgb_float_for_plot(img_to_show),
            interpolation="nearest",
        )
        ax.set_title(col["title"], fontsize=10)
        ax.axis("off")

        # Row 1: Tone curve
        ax = axes[1, col_idx]

        ax.plot(
            x,
            x,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.7,
        )

        if not is_original:
            if col["adaptive"]:
                y = sunlight_curve(
                    x,
                    p=col["p"],
                    s=col["s"],
                    **col["curve_kwargs"],
                )
            else:
                y = sunlight_curve(x, p=col["p"], s=col["s"])

            ax.plot(
                x,
                y,
                linewidth=1.7,
                color=col["curve_color"],
            )

            s_low_val = col.get("curve_kwargs", {}).get("s_low", col["s"])
            ax.text(
                0.05,
                0.92,
                f"p={col['p']:.2f}\ns_h={col['s']:.2f}\ns_l={s_low_val:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                fontfamily="monospace",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.set_title("Tone Curve", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Input", fontsize=8)
        ax.set_ylabel("Output", fontsize=8)
        ax.grid(True, linestyle="-", alpha=0.35)
        ax.tick_params(labelsize=7)

        # Row 2: Y/V histogram
        ax = axes[2, col_idx]

        if mode == "hsv":
            import cv2
            if is_original:
                _img = col["image"]
            else:
                _img = col["stage2_img"]
            _float = _img.astype(np.float32) / 255.0
            _hsv = cv2.cvtColor(_float, cv2.COLOR_BGR2HSV)
            hist_vals = _hsv[:, :, 2].reshape(-1)
            hist_title_str = "V Histogram"
            hist_xlabel_str = "V"
        else:
            if is_original:
                hist_vals = extract_y_from_image_float(col["image"], standard="bt709").reshape(-1)
            else:
                hist_vals = extract_y_from_image_float(col["stage2_img"], standard="bt709").reshape(-1)
            hist_title_str = "Y Histogram"
            hist_xlabel_str = "Luma"

        ax.hist(
            hist_vals,
            bins=32,
            range=(0.0, 1.0),
            density=True,
            alpha=0.9,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_title(hist_title_str, fontsize=10)
        ax.set_xlabel(hist_xlabel_str, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.5, linewidth=0.5, color="#cccccc")
        ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)