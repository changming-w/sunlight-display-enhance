# processor.py

from pathlib import Path

from instrument.config import (
    get_tone_params_by_lux,
    get_wb_params_by_lux,
    lux_to_name,
)

from common.utils.image_io import collect_image_paths, imread_unicode, imwrite_unicode

from instrument.tone_curve import (
    enhance_image,
    get_effective_tone_params,
)

from instrument.tone_curve_high_weaken import enhance_image_high_weaken

from visualization.stage_compare import (
    save_stage1_comparison_plot,
    save_stage2_comparison_plot,
    save_stage3_comparison_plot,
    save_edge_pipeline_visualization_plot,
    save_adaptive_comparison_plot,
)

from common.algorithm.white_balance import WhiteBalanceShifter

from common.algorithm.edge_enhance import (
    sobel_edge_enhance_image,
    get_edge_params_by_lux,
    prepare_edge_params_for_image,
)


def _get_scaled_edge_params(lux, edge_scale=1.0):
    """
    获取 Sobel 边缘增强参数，并应用全局 amount scale。
    """
    edge_params = dict(get_edge_params_by_lux(lux))
    edge_params["amount"] = float(edge_params["amount"]) * float(edge_scale)
    return edge_params


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


def process_folder(
    input_dir,
    output_dir,
    lux_list,
    mode="ycbcr",
    recursive=False,
    overwrite=True,
    save_compare=True,
    save_edge=True,
    edge_scale=1.0,

    # 内容自适应
    tone_content_adaptive=True,
    edge_content_adaptive=True,

    # Stage3 WB
    enable_wb=False,
    wb_preserve_y=True,
    wb_transfer="gamma",
    wb_gamma=2.2,

    # p_adj 自适应方法
    p_method="fixed",

    # 处理顺序
    edge_first=False,

    # 高亮弱化：None 用原版曲线；float (0~1) 启用 tone_curve_high_weaken
    high_weaken=None,
    high_weaken_start=None,       # 默认=None，从交点 p 开始
    high_weaken_end=1.0,          # 默认=1.0，到端点结束
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    image_paths = collect_image_paths(input_dir, recursive=recursive)

    if len(image_paths) == 0:
        print(f"No image found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Stage1 mode: {mode}")
    print(f"Stage1 content adaptive: {tone_content_adaptive}")
    print(f"Stage1 p_adj method: {p_method}")
    print(f"Save Stage2 Sobel edge: {save_edge}")
    print(f"Stage2 content adaptive: {edge_content_adaptive}")
    print(f"Enable Stage3 WB drift: {enable_wb}")
    print(f"Edge amount scale: {edge_scale}")
    print(f"Edge first (before tone): {edge_first}")
    if high_weaken is not None:
        start_str = f"{high_weaken_start * 255:.0f}" if high_weaken_start is not None else "p"
        print(
            f"Stage1 high weaken: {high_weaken:.2f} "
            f"(start={start_str}, end={high_weaken_end * 255:.0f})"
        )
    else:
        print("Stage1 high weaken: off (original curve)")
    print(f"Lux list: {lux_list}")

    print()
    print("Stage1 tone curve base parameters:")
    for lux in lux_list:
        p, s = get_tone_params_by_lux(lux)
        print(f"  {lux_to_name(lux)}: p = {p:.4f}, s = {s:.4f}")

    if save_edge:
        print()
        print("Stage2 Sobel edge base parameters:")
        for lux in lux_list:
            edge_params = _get_scaled_edge_params(lux, edge_scale=edge_scale)
            print(
                f"  {lux_to_name(lux)}: "
                f"amount = {edge_params['amount']:.4f}, "
                f"outline = {edge_params.get('outline_strength', 0.0):.4f}, "
                f"sigma = {edge_params['sigma']:.4f}, "
                f"edge_low = {edge_params['edge_low']:.4f}, "
                f"edge_high = {edge_params['edge_high']:.4f}"
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
    print("Start processing images")
    print("=" * 70)

    wb_shifter_cache = {}

    for idx, img_path in enumerate(image_paths):
        rel_path = img_path.relative_to(input_dir)

        print()
        print(f"[Image {idx + 1:04d}/{len(image_paths):04d}] {img_path}")

        original_img = imread_unicode(img_path)

        if original_img is None:
            print(f"  [Failed] Read image: {img_path}")
            continue

        records = []

        for lux in lux_list:
            lux_name = lux_to_name(lux)

            # =================================================
            # Stage1：Y 通道 tone curve 增强
            # =================================================
            p_base, s_base = get_tone_params_by_lux(lux)

            try:
                p_eff, s_eff, curve_kwargs, tone_stats = get_effective_tone_params(
                    original_img,
                    p_base,
                    s_base,
                    content_adaptive=tone_content_adaptive,
                    standard="bt709",
                    p_method=p_method,
                    mode=mode,
                )
            except Exception as e:
                print(f"  [Failed] Stage1 param adapt {lux_name}, error: {e}")
                continue

            # =================================================
            # edge_first: 先锐化 original，再 tone
            # =================================================
            if edge_first and save_edge:
                # 先对 original 做 edge enhance
                edge_params_base = _get_scaled_edge_params(lux, edge_scale=edge_scale)
                edge_params_eff = dict(edge_params_base)
                edge_stats = None

                try:
                    edge_params_eff, edge_stats = prepare_edge_params_for_image(
                        original_img,
                        edge_params_base,
                        standard="bt709",
                        content_adaptive=edge_content_adaptive,
                    )
                except Exception as e:
                    print(f"  [Warning] Edge param adapt {lux_name}, error: {e}")
                    edge_params_eff = dict(edge_params_base)
                    edge_stats = None

                try:
                    edge_img = sobel_edge_enhance_image(
                        original_img,
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
                    print(f"  [Failed] Edge {lux_name}, error: {e}")
                    edge_img = original_img

                # 对 edge 结果做 tone curve
                stage1_path = output_dir / "stage1_tone" / lux_name / rel_path
                try:
                    if high_weaken is not None:
                        stage1_img = enhance_image_high_weaken(
                            edge_img,
                            p=p_eff,
                            s=s_eff,
                            high_weaken=high_weaken,
                            high_weaken_start=high_weaken_start,
                            high_weaken_end=high_weaken_end,
                            curve_kwargs=curve_kwargs,
                            lux=lux,
                        )
                    else:
                        stage1_img = enhance_image(
                            edge_img,
                            p=p_eff,
                            s=s_eff,
                            mode=mode,
                            content_adaptive=False,
                            curve_kwargs=curve_kwargs,
                            lux=lux,
                        )
                except Exception as e:
                    print(f"  [Failed] Tone {lux_name}, error: {e}")
                    continue

                imwrite_unicode(stage1_path, stage1_img)
                s_low_val = curve_kwargs.get("s_low", s_eff)
                print(
                    f"  [Save] Edge->Tone {lux_name} -> {stage1_path} "
                    f"(p={p_eff:.4f}, s_h={s_eff:.4f}, s_l={s_low_val:.4f})"
                )

                # stage2 output = stage1 (已包含 edge + tone)
                stage2_path = output_dir / "stage2_sobel" / lux_name / rel_path
                imwrite_unicode(stage2_path, stage1_img)
                print(
                    f"  [Save] Stage2 {lux_name} -> {stage2_path} "
                    f"(amount={edge_params_eff['amount']:.4f}, "
                    f"sigma={edge_params_eff['sigma']:.4f})"
                )

                stage2_img = stage1_img

            else:
                # =================================================
                # 默认顺序：先 tone 再 edge
                # =================================================
                stage1_path = output_dir / "stage1_tone" / lux_name / rel_path

                if stage1_path.exists() and not overwrite:
                    print(f"  [Skip] Stage1 exists: {stage1_path}")
                    stage1_img = imread_unicode(stage1_path)

                    if stage1_img is None:
                        print(f"  [Warning] Failed to read existing Stage1: {stage1_path}")
                        continue
                else:
                    try:
                        if high_weaken is not None:
                            stage1_img = enhance_image_high_weaken(
                                original_img,
                                p=p_eff,
                                s=s_eff,
                                high_weaken=high_weaken,
                                high_weaken_start=high_weaken_start,
                                high_weaken_end=high_weaken_end,
                                curve_kwargs=curve_kwargs,
                                lux=lux,
                            )
                        else:
                            stage1_img = enhance_image(
                                original_img,
                                p=p_eff,
                                s=s_eff,
                                mode=mode,
                                content_adaptive=False,
                                curve_kwargs=curve_kwargs,
                                lux=lux,
                            )
                    except Exception as e:
                        print(f"  [Failed] Stage1 {lux_name}, error: {e}")
                        continue

                    imwrite_unicode(stage1_path, stage1_img)
                    s_low_val = curve_kwargs.get("s_low", s_eff)
                    print(
                        f"  [Save] Stage1 {lux_name} -> {stage1_path} "
                        f"(p={p_eff:.4f}, s_h={s_eff:.4f}, s_l={s_low_val:.4f})"
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

                    # 实际生效参数，保持旧字段兼容可视化
                    "p": p_eff,
                    "s": s_eff,

                    # 基础 preset 参数
                    "p_base": p_base,
                    "s_base": s_base,

                    "curve_kwargs": curve_kwargs,
                    "tone_stats": tone_stats,

                    # 高亮弱化信息（可视化用）
                    "high_weaken": high_weaken,
                    "high_weaken_start": high_weaken_start,
                    "high_weaken_end": high_weaken_end,

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
                    save_stage1_comparison_plot(
                        original_img=original_img,
                        records=records,
                        save_path=stage1_compare_path,
                        mode=mode,
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

            adaptive_compare_path = (
                output_dir
                / "_compare_adaptive"
                / rel_path.with_suffix(".adaptive_compare.png")
            )

            if adaptive_compare_path.exists() and not overwrite:
                print(f"  [Skip] Adaptive compare exists: {adaptive_compare_path}")
            else:
                try:
                    save_adaptive_comparison_plot(
                        original_img=original_img,
                        save_path=adaptive_compare_path,
                        mode=mode,
                    )
                    print(f"  [Save] Adaptive compare -> {adaptive_compare_path}")
                except Exception as e:
                    print(
                        f"  [Failed] Save Adaptive compare: "
                        f"{adaptive_compare_path}, error: {e}"
                    )