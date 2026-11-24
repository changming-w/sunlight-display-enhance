"""
Edge Enhancement V1 vs V4 — 分文件夹对比输出
V1: Original Sobel USM (EDGE_PRESETS)
V4: Content-Aware (structure tensor + asymmetric + selective dark outline)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from instrument.config import EDGE_PRESETS, get_tone_params_by_lux
from common.algorithm.edge_enhance import sobel_edge_enhance_image
from common.algorithm.content_aware_edge import (
    content_aware_edge_enhance_image,
    get_content_aware_params_by_lux,
)
from common.utils.image_io import imread_unicode, imwrite_unicode, collect_image_paths
from instrument.tone_curve import enhance_image, get_effective_tone_params


INPUT_DIR = Path("/mnt/data/wcm/sunlight_enhance/test/input")
OUTPUT_V1 = Path("/mnt/data/wcm/sunlight_enhance/test/output/edge_v1")
OUTPUT_V4 = Path("/mnt/data/wcm/sunlight_enhance/test/output/edge_v4")

OUTPUT_V1.mkdir(parents=True, exist_ok=True)
OUTPUT_V4.mkdir(parents=True, exist_ok=True)

LUX = 80000


def get_preset_params(presets, lux):
    preset_lux = sorted(presets.keys())
    if lux in presets:
        return dict(presets[lux])
    if lux >= preset_lux[-1]:
        return dict(presets[preset_lux[-1]])
    if lux <= preset_lux[0]:
        return dict(presets[preset_lux[0]])
    for i in range(len(preset_lux) - 1):
        if preset_lux[i] <= lux <= preset_lux[i+1]:
            return dict(presets[preset_lux[i+1]])
    return dict(presets[preset_lux[-1]])


def apply_stage1(img, lux):
    p_base, s_base = get_tone_params_by_lux(lux)
    p_eff, s_eff, curve_kwargs, _ = get_effective_tone_params(
        img, p_base, s_base,
        content_adaptive=True,
        standard="bt709",
        p_method="centroid",
    )
    return enhance_image(
        img, p=p_eff, s=s_eff, mode="ycbcr",
        content_adaptive=False, curve_kwargs=curve_kwargs, lux=lux,
    )


def apply_edge_v1(img, params):
    return sobel_edge_enhance_image(
        img,
        amount=params["amount"],
        sigma=params["sigma"],
        outline_strength=params.get("outline_strength", 0.0),
        edge_low=params["edge_low"],
        edge_high=params["edge_high"],
        edge_blur=params["edge_blur"],
        luma_low=params["luma_low"],
        luma_high=params["luma_high"],
        norm_percentile=params["norm_percentile"],
        standard="bt709",
        content_adaptive=False,
    )


def main():
    image_paths = collect_image_paths(INPUT_DIR, recursive=False)
    print(f"Found {len(image_paths)} images in {INPUT_DIR}")
    print(f"Lux: {LUX}")

    v1_params = get_preset_params(EDGE_PRESETS, LUX)
    v4_params = get_content_aware_params_by_lux(LUX)

    print(f"\nV1 params: amount={v1_params['amount']}, outline={v1_params.get('outline_strength', 0)}, sigma={v1_params['sigma']}")
    print(f"V4 params: sigma={v4_params['sigma']}, lambda_bright={v4_params['lambda_bright']}, lambda_dark={v4_params['lambda_dark']}, outline_dark={v4_params['outline_dark']}")
    print()

    for idx, img_path in enumerate(image_paths):
        img = imread_unicode(img_path)
        if img is None:
            print(f"[Skip] Cannot read: {img_path}")
            continue

        filename = img_path.name
        stage1 = apply_stage1(img, LUX)

        # V1
        v1_result = apply_edge_v1(stage1, v1_params)
        imwrite_unicode(OUTPUT_V1 / filename, v1_result)

        # V4
        v4_result = content_aware_edge_enhance_image(
            stage1,
            sigma=v4_params.get("sigma", 6.0),
            lambda_bright=v4_params.get("lambda_bright", 2.0),
            lambda_dark=v4_params.get("lambda_dark", 1.0),
            outline_dark=v4_params.get("outline_dark", 0.5),
            edge_low=v4_params.get("edge_low", 0.02),
            edge_high=v4_params.get("edge_high", 0.10),
            coherence_low=v4_params.get("coherence_low", 0.3),
            coherence_high=v4_params.get("coherence_high", 0.7),
            luma_low=v4_params.get("luma_low", 0.02),
            luma_high=v4_params.get("luma_high", 0.10),
            sigma_st=v4_params.get("sigma_st", 2.0),
            textlike_win=int(v4_params.get("textlike_win", 11)),
            norm_percentile=v4_params.get("norm_percentile", 99.5),
            standard="bt709",
        )
        imwrite_unicode(OUTPUT_V4 / filename, v4_result)

        print(f"[{idx+1:03d}/{len(image_paths):03d}] {filename} -> V1 & V4 saved")

    print("\nDone!")


if __name__ == "__main__":
    main()
