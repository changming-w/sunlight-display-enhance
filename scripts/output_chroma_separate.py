#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
output_chroma_separate.py

将 YCbCr 与 HSV 两种色彩补偿方案的增强结果分别输出到独立目录，便于对比。

输出目录：
    test/output8/chroma_compare/original/{filename}
    test/output8/chroma_compare/ycbcr/{filename}
    test/output8/chroma_compare/hsv/{filename}
"""

import os
import sys
import glob
import time

import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from common.utils.image_io import imread_unicode, imwrite_unicode
from common.utils.image_utils import to_float01, restore_dtype
from common.algorithm.curve_math import sunlight_curve
from instrument.tone_curve import apply_curve_hsv
from instrument.config import get_tone_params_by_lux
from instrument.tone_curve import (
    get_effective_tone_params,
    apply_curve_y_only_bgr,
)


# ============================================================
# Configuration
# ============================================================
LUX = 80000
INPUT_DIR = os.path.join(PROJECT_ROOT, "test", "input")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "test", "output8", "chroma_compare")
OUT_ORIGINAL = os.path.join(OUTPUT_BASE, "original")
OUT_YCBCR = os.path.join(OUTPUT_BASE, "ycbcr")
OUT_HSV = os.path.join(OUTPUT_BASE, "hsv")


def ensure_dirs():
    for d in (OUT_ORIGINAL, OUT_YCBCR, OUT_HSV):
        os.makedirs(d, exist_ok=True)


def process_one(img_path, lux=LUX):
    """处理单张图像，返回 (original_bgr_u8, ycbcr_bgr_u8, hsv_bgr_u8)。"""
    img = imread_unicode(img_path)
    if img is None:
        return None, None, None

    # 若为 4 通道则丢弃 alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # 获取基于 lux 的初始 tone 参数 + 内容自适应调整
    p_base, s_base = get_tone_params_by_lux(lux)
    p_eff, s_eff, curve_kwargs, _ = get_effective_tone_params(
        img, p_base, s_base,
        content_adaptive=True, standard="bt709", p_method="iqr",
    )

    # ---------- YCbCr 方案（当前主流水线） ----------
    result_ycbcr = apply_curve_y_only_bgr(
        img, p=p_eff, s=s_eff, standard="bt709",
        curve_kwargs=curve_kwargs, lux=lux,
    )

    # ---------- HSV 方案 ----------
    img_float = to_float01(img)
    _kw = dict(curve_kwargs)

    def curve_func(V_channel, _p=p_eff, _s=s_eff, _kw=_kw):
        return sunlight_curve(V_channel, p=_p, s=_s, **_kw)

    result_hsv_float = apply_curve_hsv(img_float, curve_func)
    result_hsv = restore_dtype(result_hsv_float, np.uint8)

    return img, result_ycbcr, result_hsv


def main():
    ensure_dirs()

    image_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.png"))
        + glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
    )
    if not image_paths:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} images in {INPUT_DIR}")
    print(f"[INFO] LUX = {LUX}")
    print(f"[INFO] Output base: {OUTPUT_BASE}")
    print("=" * 60)

    success = 0
    t0 = time.time()
    for i, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        print(f"[{i+1:02d}/{len(image_paths):02d}] {fname} ... ", end="", flush=True)

        original, ycbcr, hsv = process_one(img_path, lux=LUX)
        if original is None:
            print("SKIP (read failed)")
            continue

        imwrite_unicode(os.path.join(OUT_ORIGINAL, fname), original)
        imwrite_unicode(os.path.join(OUT_YCBCR, fname), ycbcr)
        imwrite_unicode(os.path.join(OUT_HSV, fname), hsv)

        success += 1
        print("OK")

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"[DONE] Processed {success}/{len(image_paths)} images in {elapsed:.1f}s")

    # 文件计数报告
    def count(d):
        if not os.path.isdir(d):
            return 0
        return sum(
            1 for n in os.listdir(d)
            if os.path.isfile(os.path.join(d, n))
        )

    print(f"[COUNT] {OUT_ORIGINAL}: {count(OUT_ORIGINAL)} files")
    print(f"[COUNT] {OUT_YCBCR}:    {count(OUT_YCBCR)} files")
    print(f"[COUNT] {OUT_HSV}:      {count(OUT_HSV)} files")


if __name__ == "__main__":
    main()
