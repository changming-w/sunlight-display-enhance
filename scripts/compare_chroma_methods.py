#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_chroma_methods.py

对比两种色彩补偿方案：
    A) YCbCr power-law chroma compensation（apply_curve_y_only_bgr）
    B) HSV 方案：HSV 空间仅对 V 通道应用 tone curve，H/S 不变（apply_curve_hsv）

对 test/input/ 下所有图像：
    - 使用相同的 tone curve 参数（lux=80000, content adaptive）
    - 分别用 A/B 方案增强
    - 生成对比合成图（上排: Original | YCbCr | HSV）

输出目录：test/output8/chroma_compare/
"""

import os
import sys
import glob
import time

import cv2
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from common.utils.image_utils import (
    get_dtype_max,
    to_float01,
    restore_dtype,
    bgr_to_ycbcr_float,
    ycbcr_to_bgr_float,
    image_to_rgb_float_for_plot,
)
from common.algorithm.curve_math import sunlight_curve
from common.algorithm.chroma_comp import (
    compensate_chroma,
    is_chroma_compensation_enabled,
)
from instrument.config import (
    get_tone_params_by_lux,
    get_chroma_compensation_by_lux,
)
from instrument.tone_curve import (
    get_effective_tone_params,
    apply_curve_y_only_bgr,
    apply_curve_hsv,
)


# ============================================================
# Configuration
# ============================================================
LUX = 80000
INPUT_DIR = os.path.join(PROJECT_ROOT, "test", "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test", "output8", "chroma_compare")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def add_label(img, text, position="top-left", font_scale=0.8, thickness=2):
    """在图像上添加文字标签。"""
    img_out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == "top-left":
        x, y = 10, th + 10
    elif position == "top-center":
        x, y = (img.shape[1] - tw) // 2, th + 10
    else:
        x, y = 10, th + 10

    # 半透明背景
    cv2.rectangle(img_out, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), bg_color, -1)
    cv2.putText(img_out, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return img_out


def compute_diff_map(img_a, img_b, amplify=5.0):
    """
    计算两张图像的差异图（放大显示）。
    输入 uint8 BGR，输出 uint8 BGR 差异热力图。
    """
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    diff = np.abs(a - b)
    diff_gray = np.mean(diff, axis=2)
    diff_amplified = np.clip(diff_gray * amplify, 0, 255).astype(np.uint8)
    # 使用 colormap 可视化
    heatmap = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)
    return heatmap


def process_image(img_path, lux=LUX):
    """
    处理单张图像，返回 (original, ycbcr_result, hsv_result, stats_dict)。
    全部以 uint8 BGR 格式返回。
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None, None

    # 获取基于 lux 的初始 tone 参数
    p_base, s_base = get_tone_params_by_lux(lux)

    # 内容自适应调整参数
    p_eff, s_eff, curve_kwargs, tone_stats = get_effective_tone_params(
        img, p_base, s_base, content_adaptive=True, standard="bt709", p_method="iqr"
    )

    # ============================================================
    # 方案 A: YCbCr + power-law chroma compensation（当前方案）
    # ============================================================
    result_ycbcr = apply_curve_y_only_bgr(
        img, p=p_eff, s=s_eff, standard="bt709",
        curve_kwargs=curve_kwargs, lux=lux,
    )

    # ============================================================
    # 方案 B: HSV V-channel tone curve
    # ============================================================
    img_float = to_float01(img)

    def _curve_for_hsv(V_channel, _p=p_eff, _s=s_eff, _kw=dict(curve_kwargs)):
        return sunlight_curve(V_channel, p=_p, s=_s, **_kw)

    result_hsv_float = apply_curve_hsv(img_float, _curve_for_hsv)
    result_hsv = restore_dtype(result_hsv_float, np.uint8)

    # ============================================================
    # 饱和度统计（HSV S 通道均值）
    # ============================================================
    def _mean_sat(bgr_u8):
        f = bgr_u8.astype(np.float32) / 255.0
        return float(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean())

    stats = {
        "p_eff": p_eff,
        "s_eff": s_eff,
        "sat_orig": _mean_sat(img),
        "sat_ycbcr": _mean_sat(result_ycbcr),
        "sat_hsv": _mean_sat(result_hsv),
    }

    return img, result_ycbcr, result_hsv, stats


def build_composite(original, ycbcr, hsv, name, stats):
    """
    构建对比合成图。
    上排：Original | YCbCr | HSV
    下排：Diff(Orig vs YCbCr) | Diff(Orig vs HSV) | Diff(YCbCr vs HSV)
    """
    h, w = original.shape[:2]

    # 统一尺寸（取最大高宽对齐）
    target_h = h
    target_w = w

    def resize_if_needed(img):
        if img.shape[0] != target_h or img.shape[1] != target_w:
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return img

    orig = resize_if_needed(original)
    yc = resize_if_needed(ycbcr)
    hv = resize_if_needed(hsv)

    # 添加标签
    orig_labeled = add_label(
        orig, f"Original  S={stats['sat_orig']:.3f}"
    )
    ycbcr_labeled = add_label(
        yc,
        f"YCbCr (p={stats['p_eff']:.2f}, s={stats['s_eff']:.2f})  "
        f"S={stats['sat_ycbcr']:.3f}",
    )
    hsv_labeled = add_label(
        hv,
        f"HSV (V-only)  S={stats['sat_hsv']:.3f}",
    )

    # 差异图
    diff_orig_yc = add_label(
        compute_diff_map(orig, yc), "Diff: Orig vs YCbCr", font_scale=0.6
    )
    diff_orig_hv = add_label(
        compute_diff_map(orig, hv), "Diff: Orig vs HSV", font_scale=0.6
    )
    diff_yc_hv = add_label(
        compute_diff_map(yc, hv), "Diff: YCbCr vs HSV", font_scale=0.6
    )

    # 合成
    top_row = np.hstack([orig_labeled, ycbcr_labeled, hsv_labeled])
    bottom_row = np.hstack([diff_orig_yc, diff_orig_hv, diff_yc_hv])
    composite = np.vstack([top_row, bottom_row])

    return composite


def main():
    ensure_dir(OUTPUT_DIR)

    # 查找输入图像
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not image_paths:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} test images in {INPUT_DIR}")
    print(f"[INFO] LUX = {LUX}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    success_count = 0
    t_start = time.time()

    for i, img_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1:02d}/{len(image_paths):02d}] Processing: {name} ... ", end="", flush=True)

        original, ycbcr, hsv, stats = process_image(img_path, lux=LUX)

        if original is None:
            print("SKIP (cannot read)")
            continue

        composite = build_composite(original, ycbcr, hsv, name, stats)

        out_path = os.path.join(OUTPUT_DIR, f"{name}_chroma_compare.png")
        cv2.imwrite(out_path, composite)

        success_count += 1

        print(
            f"OK  ("
            f"S: orig={stats['sat_orig']:.3f} "
            f"yc={stats['sat_ycbcr']:.3f} "
            f"hsv={stats['sat_hsv']:.3f})"
        )

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"[DONE] Processed {success_count}/{len(image_paths)} images in {elapsed:.1f}s")
    print(f"[DONE] Output: {OUTPUT_DIR} ({success_count} files)")


if __name__ == "__main__":
    main()
