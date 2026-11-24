"""
对比实验：HSV 间接 vs HSV 直接 vs YCbCr

三种方式：
  1. hsv_indirect: 当前方式 — 基于 Y 计算 gain，乘到 V
  2. hsv_direct:   新方式 — 曲线直接作用在 V 上
  3. ycbcr:        参照 — 曲线作用在 Y 上 + 色度补偿

Usage:
    python scripts/compare_hsv_methods.py --input ./test/input1 --output ./test/compare_hsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.utils.image_io import collect_image_paths, imread_unicode, imwrite_unicode
from instrument.tone_curve import (
    enhance_image,
    get_effective_tone_params,
    sunlight_curve,
)
from instrument.config import LUX_PRESETS


def apply_curve_hsv_direct(img_bgr_float, curve_func):
    """HSV 直接方式：曲线直接作用在 V 通道上，H/S 不变。"""
    hsv = cv2.cvtColor(img_bgr_float.astype(np.float32), cv2.COLOR_BGR2HSV)
    V_old = hsv[:, :, 2]
    V_new = curve_func(V_old)
    hsv[:, :, 2] = np.clip(V_new, 0.0, 1.0)
    bgr_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(bgr_new, 0.0, 1.0).astype(np.float32)


def get_dtype_max(img):
    if img.dtype == np.uint8:
        return 255.0
    elif img.dtype == np.uint16:
        return 65535.0
    return 1.0


def restore_dtype(img_float, dtype):
    if dtype == np.uint8:
        return np.clip(img_float * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif dtype == np.uint16:
        return np.clip(img_float * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    return img_float


def main():
    parser = argparse.ArgumentParser(description="Compare HSV indirect vs direct vs YCbCr")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lux", type=float, default=80000)
    args = parser.parse_args()

    input_dir = args.input
    output_root = args.output
    lux = args.lux

    if not input_dir.is_dir():
        raise SystemExit(f"--input is not a directory: {input_dir}")

    # 获取对应 lux 档位的 p/s 参数
    lux_keys = sorted(LUX_PRESETS.keys())
    # 找最近的 lux 档
    preset_lux = min(lux_keys, key=lambda k: abs(k - lux))
    preset = LUX_PRESETS[preset_lux]
    base_p = preset["p"]
    base_s = preset["s"]
    print(f"Using LUX_PRESETS[{preset_lux}]: p={base_p}, s={base_s}")

    paths = collect_image_paths(input_dir)
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")

    dir_ycbcr = output_root / "ycbcr"
    dir_hsv_indirect = output_root / "hsv_indirect"
    dir_hsv_direct = output_root / "hsv_direct"

    for img_path in paths:
        img = imread_unicode(img_path)
        if img is None:
            continue
        print(f"Processing {img_path.name}")

        # 内容自适应参数
        p_eff, s_eff, curve_kwargs, _ = get_effective_tone_params(
            img, base_p, base_s, content_adaptive=True, standard="bt709"
        )
        print(f"  p_eff={p_eff:.3f}, s_eff={s_eff:.3f}")

        # --- 1. YCbCr ---
        out_ycbcr, _ = enhance_image(
            img, p_eff, s_eff, mode="ycbcr",
            content_adaptive=False, curve_kwargs=curve_kwargs,
            lux=lux, return_info=True,
        )
        imwrite_unicode(dir_ycbcr / img_path.name, out_ycbcr)

        # --- 2. HSV indirect (当前) ---
        out_hsv_ind, _ = enhance_image(
            img, p_eff, s_eff, mode="hsv",
            content_adaptive=False, curve_kwargs=curve_kwargs,
            return_info=True,
        )
        imwrite_unicode(dir_hsv_indirect / img_path.name, out_hsv_ind)

        # --- 3. HSV direct (新方案) ---
        from functools import partial
        curve_func = partial(sunlight_curve, p=p_eff, s=s_eff, **curve_kwargs)

        img_float = img.astype(np.float32) / get_dtype_max(img)
        img_float = np.clip(img_float, 0.0, 1.0)
        bgr_direct = apply_curve_hsv_direct(img_float, curve_func)
        out_hsv_dir = restore_dtype(bgr_direct, img.dtype)
        imwrite_unicode(dir_hsv_direct / img_path.name, out_hsv_dir)

    print(f"\nDone. Results saved to:")
    print(f"  {dir_ycbcr}")
    print(f"  {dir_hsv_indirect}")
    print(f"  {dir_hsv_direct}")


if __name__ == "__main__":
    main()
