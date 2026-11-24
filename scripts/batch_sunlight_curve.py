import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/wcm/sunlight_display/sunlight_enhance")

import cv2
import numpy as np

from sobel_edge_enhance import (
    sobel_edge_enhance_image,
    get_edge_params_by_lux,
    compute_sobel_debug_image,
    extract_y_from_image_float,
)

# ============================================================
# 1. 预设参数：不同光照强度对应不同曲线
# ============================================================

LUX_PRESETS = {
    5000: {
        "p": 0.45,
        "s": 1.25,
    },
    25000: {
        "p": 0.42,
        "s": 1.65,
    },
    45000: {
        "p": 0.40,
        "s": 2.10,
    },
}


# ============================================================
# 2. 图像读写：支持中文路径
# ============================================================

def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """
    支持中文路径的图像读取
    """
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path, img):
    """
    支持中文路径的图像保存
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix
    if ext == "":
        ext = ".png"

    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


# ============================================================
# 3. 曲线函数
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def sunlight_curve(x, p=0.45, s=1.65):
    """
    阳光可见性增强曲线

    参数：
        x: 输入亮度，范围 [0, 1]
        p: 与 y=x 的交点
        s: 曲线强度，s=1 时退化为 y=x，s>1 时增强对比度

    返回：
        y: 输出亮度，范围 [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)

    z = s * logit(x) + (1.0 - s) * logit(p)
    y = sigmoid(z)

    # 保证端点严格映射
    y = np.where(x <= 0.0, 0.0, y)
    y = np.where(x >= 1.0, 1.0, y)

    return y.astype(np.float32)


# ============================================================
# 4. 参数插值：如果输入 lux 不在 preset 中，可线性插值
# ============================================================

def get_params_by_lux(lux):
    """
    根据 lux 获取 p 和 s。
    如果 lux 正好是 preset，直接使用。
    如果不在 preset 中，则做线性插值或边界外推。
    """
    lux = float(lux)

    preset_lux = sorted(LUX_PRESETS.keys())

    if int(lux) in LUX_PRESETS:
        item = LUX_PRESETS[int(lux)]
        return item["p"], item["s"]

    # 小于最小 lux，使用最小档
    if lux <= preset_lux[0]:
        item = LUX_PRESETS[preset_lux[0]]
        return item["p"], item["s"]

    # 大于最大 lux，使用最大档
    if lux >= preset_lux[-1]:
        item = LUX_PRESETS[preset_lux[-1]]
        return item["p"], item["s"]

    # 中间线性插值
    for i in range(len(preset_lux) - 1):
        l0 = preset_lux[i]
        l1 = preset_lux[i + 1]

        if l0 <= lux <= l1:
            t = (lux - l0) / (l1 - l0)

            p0 = LUX_PRESETS[l0]["p"]
            s0 = LUX_PRESETS[l0]["s"]

            p1 = LUX_PRESETS[l1]["p"]
            s1 = LUX_PRESETS[l1]["s"]

            p = p0 * (1.0 - t) + p1 * t
            s = s0 * (1.0 - t) + s1 * t

            return p, s

    raise RuntimeError("Unexpected lux interpolation error.")


# ============================================================
# 5. 图像 bit depth 判断
# ============================================================

def get_dtype_max(img):
    """
    获取图像数据类型对应的最大值
    """
    if img.dtype == np.uint8:
        return 255.0
    elif img.dtype == np.uint16:
        return 65535.0
    else:
        # 对 float 图默认认为已经是 [0, 1]
        return 1.0


def restore_dtype(img_float, dtype):
    """
    将 [0,1] float 图像转回原始 dtype
    """
    img_float = np.clip(img_float, 0.0, 1.0)

    if dtype == np.uint8:
        return np.round(img_float * 255.0).astype(np.uint8)
    elif dtype == np.uint16:
        return np.round(img_float * 65535.0).astype(np.uint16)
    else:
        return img_float.astype(dtype)


# ============================================================
# 6. 生成 LUT
# ============================================================

def build_lut(dtype, p, s):
    """
    为 uint8 / uint16 图像生成 LUT
    """
    if dtype == np.uint8:
        max_value = 255
        lut_dtype = np.uint8
    elif dtype == np.uint16:
        max_value = 65535
        lut_dtype = np.uint16
    else:
        return None

    x = np.linspace(0.0, 1.0, max_value + 1, dtype=np.float32)
    y = sunlight_curve(x, p=p, s=s)

    lut = np.round(y * max_value)
    lut = np.clip(lut, 0, max_value).astype(lut_dtype)

    return lut


# ============================================================
# 7. 图像增强：两种模式
# ============================================================

def apply_curve_per_channel(img, p, s):
    """
    对 RGB/BGR 每个通道分别做曲线。
    优点：实现简单，速度快。
    缺点：可能轻微改变色相和饱和度。
    """
    dtype = img.dtype

    lut = build_lut(dtype, p, s)

    if lut is not None:
        return lut[img]

    # float 图像
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    out = sunlight_curve(img_float, p=p, s=s)
    return out.astype(dtype)


def apply_curve_luma_preserve_color(img, p, s):
    """
    推荐方式：基于亮度 Y 做曲线，再用增益回乘 BGR/RGB。

    优点：
        - 更接近显示亮度调节
        - 相比 per-channel 更不容易产生明显 hue shift
    缺点：
        - 高饱和区域可能因为通道 clip 而略有压缩
    """
    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    # OpenCV 默认为 BGR 排列
    # BT.709 luma:
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    B = img_float[..., 0]
    G = img_float[..., 1]
    R = img_float[..., 2]

    Y = 0.0722 * B + 0.7152 * G + 0.2126 * R
    Y_new = sunlight_curve(Y, p=p, s=s)

    eps = 1e-6
    gain = Y_new / np.maximum(Y, eps)

    out = img_float * gain[..., None]
    out = np.clip(out, 0.0, 1.0)

    return restore_dtype(out, dtype)

def apply_curve_y_only_bgr(img, p, s, standard="bt709"):
    """
    只调节 Y 通道，保持色度 Cb/Cr 不变。

    输入：
        img: OpenCV BGR 图像，uint8 / uint16 / float
        p, s: 曲线参数
        standard:
            "bt709": 推荐用于高清显示/视频场景
            "bt601": 可用于传统 SD 图像

    注意：
        - 这里使用 full range YCbCr 公式；
        - 只修改 Y，不修改 Cb/Cr；
        - 最终转回 BGR 时，如果 RGB 超出 [0,1]，clip 可能会导致极高饱和区域有轻微色度变化。
    """
    dtype = img.dtype
    max_value = get_dtype_max(img)

    img_float = img.astype(np.float32) / max_value
    img_float = np.clip(img_float, 0.0, 1.0)

    B = img_float[..., 0]
    G = img_float[..., 1]
    R = img_float[..., 2]

    if standard.lower() == "bt601":
        Kr = 0.299
        Kb = 0.114
    elif standard.lower() == "bt709":
        Kr = 0.2126
        Kb = 0.0722
    else:
        raise ValueError(f"Unsupported YUV standard: {standard}")

    Kg = 1.0 - Kr - Kb

    # --------------------------------------------------------
    # RGB/BGR -> YCbCr
    # full range:
    # Y  = Kr * R + Kg * G + Kb * B
    # Cb = (B - Y) / (2 * (1 - Kb))
    # Cr = (R - Y) / (2 * (1 - Kr))
    # --------------------------------------------------------
    Y = Kr * R + Kg * G + Kb * B
    Cb = (B - Y) / (2.0 * (1.0 - Kb))
    Cr = (R - Y) / (2.0 * (1.0 - Kr))

    # --------------------------------------------------------
    # 只对 Y 做 sunlight 曲线
    # --------------------------------------------------------
    Y_new = sunlight_curve(Y, p=p, s=s)

    # --------------------------------------------------------
    # YCbCr -> RGB/BGR，Cb/Cr 不变
    # --------------------------------------------------------
    R_new = Y_new + 2.0 * (1.0 - Kr) * Cr
    B_new = Y_new + 2.0 * (1.0 - Kb) * Cb
    G_new = (Y_new - Kr * R_new - Kb * B_new) / Kg

    out = np.stack([B_new, G_new, R_new], axis=-1)
    out = np.clip(out, 0.0, 1.0)

    return restore_dtype(out, dtype)

def enhance_image(img, p, s, mode="y_only"):
    """
    对单张图片应用阳光可见性增强曲线。

    支持：
        - 灰度图
        - BGR
        - BGRA
        - uint8 / uint16

    mode:
        y_only:
            转 YCbCr，只调 Y，保持 Cb/Cr 不变，推荐当前使用。
        luma:
            根据亮度生成 gain，再回乘 BGR，近似保色，但不是严格 Y-only。
        rgb / per_channel:
            每个通道独立套曲线，可能改变色相。
    """
    if img is None:
        return None

    # 灰度图，本身只有 Y/亮度概念
    if img.ndim == 2:
        return apply_curve_per_channel(img, p, s)

    if img.ndim == 3:
        h, w, c = img.shape

        # BGR
        if c == 3:
            if mode == "rgb" or mode == "per_channel":
                return apply_curve_per_channel(img, p, s)

            elif mode == "luma":
                return apply_curve_luma_preserve_color(img, p, s)

            elif mode == "y_only" or mode == "yuv_y_only":
                return apply_curve_y_only_bgr(img, p, s, standard="bt709")

            else:
                raise ValueError(f"Unsupported mode: {mode}")

        # BGRA，保留 alpha 通道
        elif c == 4:
            bgr = img[..., :3]
            alpha = img[..., 3:4]

            if mode == "rgb" or mode == "per_channel":
                bgr_out = apply_curve_per_channel(bgr, p, s)

            elif mode == "luma":
                bgr_out = apply_curve_luma_preserve_color(bgr, p, s)

            elif mode == "y_only" or mode == "yuv_y_only":
                bgr_out = apply_curve_y_only_bgr(bgr, p, s, standard="bt709")

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            return np.concatenate([bgr_out, alpha], axis=-1)

    raise ValueError(f"Unsupported image shape: {img.shape}")

# ============================================================
# 8. 对比图生成
# ============================================================

def lux_to_name(lux):
    """
    将 lux 数值转成目录名，例如 5000 -> 5000lux
    """
    lux = float(lux)
    if abs(lux - round(lux)) < 1e-6:
        return f"{int(round(lux))}lux"
    else:
        return f"{lux:g}lux"


def image_to_rgb_float_for_plot(img):
    """
    将 OpenCV 读取的 BGR/BGRA/Gray 图像转成 matplotlib 显示用 RGB float [0,1]
    """
    if img is None:
        raise ValueError("Input image is None.")

    dtype = img.dtype
    max_value = get_dtype_max(img)

    if img.ndim == 2:
        gray = img.astype(np.float32) / max_value
        gray = np.clip(gray, 0.0, 1.0)
        rgb = np.stack([gray, gray, gray], axis=-1)
        return rgb

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]

        bgr = img.astype(np.float32) / max_value
        bgr = np.clip(bgr, 0.0, 1.0)

        # BGR -> RGB
        rgb = bgr[..., ::-1]
        return rgb

    raise ValueError(f"Unsupported image shape for plot: {img.shape}")


def compute_luma_float(img):
    """
    计算归一化 luma，范围 [0,1]
    OpenCV 输入默认为 BGR/BGRA。
    """
    if img is None:
        raise ValueError("Input image is None.")

    dtype = img.dtype
    max_value = get_dtype_max(img)

    if img.ndim == 2:
        y = img.astype(np.float32) / max_value
        return np.clip(y, 0.0, 1.0)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]

        img_float = img.astype(np.float32) / max_value
        img_float = np.clip(img_float, 0.0, 1.0)

        B = img_float[..., 0]
        G = img_float[..., 1]
        R = img_float[..., 2]

        # BT.709 luma
        Y = 0.0722 * B + 0.7152 * G + 0.2126 * R
        return np.clip(Y, 0.0, 1.0)

    raise ValueError(f"Unsupported image shape for luma: {img.shape}")

def compute_contrast_metrics(img, local_sigma=3.0):
    """
    计算图像 Y 通道对比度指标。

    返回：
        metrics = {
            "mean_255": 平均亮度，8bit尺度
            "std_255": Y标准差，8bit尺度
            "p5_255": 5%分位亮度
            "p50_255": 中位亮度
            "p95_255": 95%分位亮度
            "p95_p5_255": 有效动态范围
            "michelson_p95_p5": 分位数 Michelson 对比度
            "local_std_mean_255": 局部标准差均值
        }
    """
    Y = compute_luma_float(img).astype(np.float32)
    Y = np.clip(Y, 0.0, 1.0)

    eps = 1e-8

    mean = float(np.mean(Y))
    std = float(np.std(Y))

    p5, p50, p95 = np.percentile(Y.reshape(-1), [5, 50, 95])
    p5 = float(p5)
    p50 = float(p50)
    p95 = float(p95)

    p95_p5 = p95 - p5
    michelson = p95_p5 / max(p95 + p5, eps)

    # 局部对比度：local std
    # local_std = sqrt(E[x^2] - E[x]^2)
    mean_local = cv2.GaussianBlur(
        Y,
        ksize=(0, 0),
        sigmaX=float(local_sigma),
        sigmaY=float(local_sigma),
        borderType=cv2.BORDER_REFLECT101,
    )

    mean2_local = cv2.GaussianBlur(
        Y * Y,
        ksize=(0, 0),
        sigmaX=float(local_sigma),
        sigmaY=float(local_sigma),
        borderType=cv2.BORDER_REFLECT101,
    )

    local_var = np.maximum(mean2_local - mean_local * mean_local, 0.0)
    local_std = np.sqrt(local_var)

    local_std_mean = float(np.mean(local_std))

    metrics = {
        "mean_255": mean * 255.0,
        "std_255": std * 255.0,
        "p5_255": p5 * 255.0,
        "p50_255": p50 * 255.0,
        "p95_255": p95 * 255.0,
        "p95_p5_255": p95_p5 * 255.0,
        "michelson_p95_p5": michelson,
        "local_std_mean_255": local_std_mean * 255.0,
    }

    return metrics

def plot_stage1_contrast_metrics(ax, original_img, records):
    """
    绘制 Stage1 对比度量化柱状图。

    对比指标：
        1. Y std ratio
        2. P95-P5 ratio
        3. Local std ratio

    所有结果都相对于 Original 归一化。

    返回：
        metric_text: str，用于在右侧单独显示指标数值
    """
    orig_m = compute_contrast_metrics(original_img)

    eps = 1e-8

    lux_names = []
    std_ratios = []
    range_ratios = []
    local_ratios = []

    text_lines = []
    text_lines.append("Stage1 Contrast Metrics")
    text_lines.append("=" * 30)
    text_lines.append("")
    text_lines.append("Original")
    text_lines.append("-" * 30)
    text_lines.append(f"Y mean       : {orig_m['mean_255']:.2f}")
    text_lines.append(f"Y std        : {orig_m['std_255']:.2f}")
    text_lines.append(f"P95-P5       : {orig_m['p95_p5_255']:.2f}")
    text_lines.append(f"Local std    : {orig_m['local_std_mean_255']:.2f}")
    text_lines.append("")

    for rec in records:
        lux = rec["lux"]
        img = rec["stage1_img"]

        m = compute_contrast_metrics(img)

        std_ratio = m["std_255"] / max(orig_m["std_255"], eps)
        range_ratio = m["p95_p5_255"] / max(orig_m["p95_p5_255"], eps)
        local_ratio = m["local_std_mean_255"] / max(orig_m["local_std_mean_255"], eps)

        lux_name = lux_to_name(lux)

        lux_names.append(lux_name)
        std_ratios.append(std_ratio)
        range_ratios.append(range_ratio)
        local_ratios.append(local_ratio)

        text_lines.append(lux_name)
        text_lines.append("-" * 30)
        text_lines.append(f"Y std        : {m['std_255']:.2f}  x{std_ratio:.3f}")
        text_lines.append(f"P95-P5       : {m['p95_p5_255']:.2f}  x{range_ratio:.3f}")
        text_lines.append(f"Local std    : {m['local_std_mean_255']:.2f}  x{local_ratio:.3f}")
        text_lines.append(f"Y mean       : {m['mean_255']:.2f}")
        text_lines.append("")

    x = np.arange(len(records), dtype=np.float32)
    width = 0.24

    ax.bar(
        x - width,
        std_ratios,
        width,
        label="Y std",
        color="#1f77b4",
        alpha=0.90,
    )

    ax.bar(
        x,
        range_ratios,
        width,
        label="P95-P5",
        color="#ff7f0e",
        alpha=0.90,
    )

    ax.bar(
        x + width,
        local_ratios,
        width,
        label="Local std",
        color="#2ca02c",
        alpha=0.90,
    )

    ax.axhline(
        1.0,
        color="k",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="Original = 1.0",
    )

    all_values = std_ratios + range_ratios + local_ratios
    ymax = max(max(all_values), 1.0) * 1.25

    # 柱状图顶部只标简单倍率，不再放大段文字
    for i in range(len(records)):
        vals = [
            std_ratios[i],
            range_ratios[i],
            local_ratios[i],
        ]
        xpos = [
            x[i] - width,
            x[i],
            x[i] + width,
        ]

        for xx, vv in zip(xpos, vals):
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
    ax.set_xticklabels(lux_names, rotation=0)
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, ymax)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8)

    metric_text = "\n".join(text_lines)
    return metric_text

def save_stage1_comparison_plot(
    original_img,
    records,
    save_path,
    mode="y_only",
    hist_bins=32,
):
    """
    第一阶段对比图：
        - 第一行：Original + 不同 lux 的 Stage1 Tone 结果
        - 第二行左侧：Stage1 对比度指标柱状图
        - 第二行中间：Stage1 对比度指标数值表
        - 第二行右侧：窄版 Stage1 Tone LUT 曲线

    hist_bins 参数保留是为了兼容原有调用，这里不再使用。
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_levels = len(records)
    n_imgs = 1 + n_levels

    fig_w = 16.0
    fig_h = 7.6

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)

    # 固定 12 列，方便控制下面三个区域宽度
    gs = fig.add_gridspec(
        2,
        12,
        height_ratios=[1.0, 1.45],
        hspace=0.36,
        wspace=0.35,
    )

    # ========================================================
    # 第一行：Original + Stage1 结果
    # ========================================================
    # 将 12 列均分给 Original + 各 lux 图
    col_edges = np.linspace(0, 12, n_imgs + 1).astype(int)

    ax = fig.add_subplot(gs[0, col_edges[0]:col_edges[1]])
    ax.imshow(image_to_rgb_float_for_plot(original_img))
    ax.set_title("Original", fontsize=10)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        p = rec["p"]
        s = rec["s"]
        img = rec["stage1_img"]

        ax = fig.add_subplot(gs[0, col_edges[i + 1]:col_edges[i + 2]])
        ax.imshow(image_to_rgb_float_for_plot(img))
        ax.set_title(
            f"Stage1 Tone\n{lux_to_name(lux)}, p={p:.2f}, s={s:.2f}",
            fontsize=10,
        )
        ax.axis("off")

    # ========================================================
    # 第二行左侧：Stage1 对比度指标柱状图
    # ========================================================
    ax_metric = fig.add_subplot(gs[1, 0:4])

    metric_text = plot_stage1_contrast_metrics(
        ax=ax_metric,
        original_img=original_img,
        records=records,
    )

    # ========================================================
    # 第二行中间：指标数值表
    # ========================================================
    ax_text = fig.add_subplot(gs[1, 4:6])
    ax_text.axis("off")

    ax_text.text(
        0.02,
        0.98,
        metric_text,
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        transform=ax_text.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95,
        ),
    )

    ax_text.set_title("Metric Values", fontsize=10)

    # ========================================================
    # 第二行右侧：窄版 Stage1 LUT 曲线
    # ========================================================
    ax_curve = fig.add_subplot(gs[1, 7:12])

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
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    for i, rec in enumerate(records):
        lux = rec["lux"]
        p = rec["p"]
        s = rec["s"]

        y = sunlight_curve(x, p=p, s=s)
        y_8bit = np.clip(np.round(y * 255.0), 0, 255)

        color = colors[i % len(colors)]

        ax_curve.plot(
            x_8bit,
            y_8bit,
            linewidth=1.7,
            color=color,
            label=f"{lux_to_name(lux)}",
        )

        ax_curve.scatter(
            [p * 255.0],
            [p * 255.0],
            s=18,
            color=color,
            zorder=5,
        )

    ax_curve.set_title(f"Tone LUTs, mode={mode}", fontsize=10)
    ax_curve.set_xlim(0, 255)
    ax_curve.set_ylim(0, 255)
    ax_curve.set_xlabel("Input")
    ax_curve.set_ylabel("Output")
    ax_curve.grid(True, linestyle="-", alpha=0.35)
    ax_curve.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_stage2_comparison_plot(
    original_img,
    records,
    save_path,
    hist_bins=32,
):
    """
    第二阶段极简对比图：

        第一行：
            Original + Stage2 Final Outputs

        第二行：
            Edge Weight Maps，黑白显示

        第三行：
            |Delta Y| Maps，黑白显示

    删除：
        - Stage1 输入过程
        - 曲线图
        - 直方图
        - 文字指标表
        - 复杂指标标题

    只保留关键视觉图。
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_levels = len(records)

    # 图像放大一些
    fig_w = max(15.5, 4.2 * (n_levels + 1))
    fig_h = 9.2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # 使用三段式布局：
    # 第一行：Original + Final，n_levels + 1 列
    # 第二行：Edge Weight，n_levels 列
    # 第三行：Delta Y，n_levels 列
    outer_gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.05, 1.0, 1.0],
        hspace=0.18,
    )

    gs_top = outer_gs[0].subgridspec(
        1,
        n_levels + 1,
        wspace=0.08,
    )

    gs_edge = outer_gs[1].subgridspec(
        1,
        n_levels,
        wspace=0.08,
    )

    gs_delta = outer_gs[2].subgridspec(
        1,
        n_levels,
        wspace=0.08,
    )

    # ========================================================
    # 预先计算 Edge Weight 和 Delta Y
    # ========================================================
    debug_list = []

    for rec in records:
        stage1_img = rec["stage1_img"]
        stage2_img = rec["stage2_img"]
        edge_params = rec["edge_params"]

        debug = compute_sobel_debug_image(
            stage1_img,
            amount=edge_params["amount"],
            sigma=edge_params["sigma"],
            edge_low=edge_params["edge_low"],
            edge_high=edge_params["edge_high"],
            edge_blur=edge_params["edge_blur"],
            luma_low=edge_params["luma_low"],
            luma_high=edge_params["luma_high"],
            norm_percentile=edge_params["norm_percentile"],
            standard="bt709",
        )

        Y1 = extract_y_from_image_float(stage1_img, standard="bt709")
        Y2 = extract_y_from_image_float(stage2_img, standard="bt709")

        delta_y = np.abs(Y2 - Y1)

        debug["delta_y"] = delta_y
        debug_list.append(debug)

    # 统一 DeltaY 显示范围，方便不同 lux 横向比较
    if len(debug_list) > 0:
        all_delta = np.concatenate(
            [d["delta_y"].reshape(-1) for d in debug_list],
            axis=0,
        )
        delta_vmax = float(np.percentile(all_delta, 99))
        delta_vmax = max(delta_vmax, 1.0 / 255.0)
    else:
        delta_vmax = 1.0 / 255.0

    # ========================================================
    # 第一行：Original + Stage2 Final
    # ========================================================
    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(image_to_rgb_float_for_plot(original_img))
    ax.set_title("Original", fontsize=11)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        stage2_img = rec["stage2_img"]

        ax = fig.add_subplot(gs_top[0, i + 1])
        ax.imshow(image_to_rgb_float_for_plot(stage2_img))
        ax.set_title(
            f"Stage2 Final {lux_to_name(lux)}",
            fontsize=11,
        )
        ax.axis("off")

    # ========================================================
    # 第二行：Edge Weight Map，黑白
    # ========================================================
    for i, rec in enumerate(records):
        lux = rec["lux"]
        edge_weight = debug_list[i]["edge_weight"]

        ax = fig.add_subplot(gs_edge[0, i])
        ax.imshow(
            edge_weight,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(
            f"Edge Weight {lux_to_name(lux)}",
            fontsize=11,
        )
        ax.axis("off")

    # ========================================================
    # 第三行：|Delta Y| Map，黑白
    # ========================================================
    for i, rec in enumerate(records):
        lux = rec["lux"]
        delta_y = debug_list[i]["delta_y"]

        ax = fig.add_subplot(gs_delta[0, i])
        ax.imshow(
            delta_y,
            cmap="gray",
            vmin=0.0,
            vmax=delta_vmax,
        )
        ax.set_title(
            f"|Delta Y| {lux_to_name(lux)}",
            fontsize=11,
        )
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def enhanced_detail_signed_to_vis(
    enhanced_detail,
    percentile=99.5,
):
    """
    signed enhanced detail 可视化范围。

    显示规则：
        红色：enhanced_detail > 0，加亮
        蓝色：enhanced_detail < 0，压暗
        接近白色/浅灰：enhanced_detail ≈ 0

    注意：
        这里不做 gain 放大，只用 percentile 做对称显示范围，
        避免极少数异常值把整体对比度压得看不见。
    """
    x = enhanced_detail.astype(np.float32)

    vmax = float(np.percentile(np.abs(x).reshape(-1), percentile))
    vmax = max(vmax, 1e-8)

    return x, vmax


def save_edge_pipeline_visualization_plot(
    records,
    save_path,
):
    """
    保存边缘增强流程图，极简版。

    只显示最高 lux 那一组，并且只放一行三张图：

        RGB Input -> Edge Weight -> Final Output

    删除：
        - Y Channel
        - Enhanced Detail
        - 低 lux / 中 lux 行
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if records is None or len(records) <= 0:
        return

    # ========================================================
    # 只取最高光照强度那一组
    # ========================================================
    rec = max(records, key=lambda r: float(r["lux"]))

    lux = rec["lux"]
    stage1_img = rec["stage1_img"]
    stage2_img = rec["stage2_img"]
    edge_params = rec["edge_params"]

    # ========================================================
    # 1. RGB Input，也就是进入边缘增强模块的图
    # ========================================================
    rgb_input_vis = image_to_rgb_float_for_plot(stage1_img)

    # ========================================================
    # 2. 计算 Edge Weight
    # ========================================================
    debug = compute_sobel_debug_image(
        stage1_img,
        amount=edge_params["amount"],
        sigma=edge_params["sigma"],
        edge_low=edge_params["edge_low"],
        edge_high=edge_params["edge_high"],
        edge_blur=edge_params["edge_blur"],
        luma_low=edge_params["luma_low"],
        luma_high=edge_params["luma_high"],
        norm_percentile=edge_params["norm_percentile"],
        standard="bt709",
    )

    edge_weight = debug["edge_weight"]

    # ========================================================
    # 3. Final Output
    # ========================================================
    final_vis = image_to_rgb_float_for_plot(stage2_img)

    # ========================================================
    # 绘图：一行三张
    # ========================================================
    fig_w = 14.0
    fig_h = 4.4

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        dpi=180,
        squeeze=False,
    )

    axes = axes[0]

    # --------------------------------------------------------
    # RGB Input
    # --------------------------------------------------------
    ax = axes[0]
    ax.imshow(
        rgb_input_vis,
        interpolation="nearest",
    )
    ax.axis("off")
    ax.set_title(
        f"RGB Input\n{lux_to_name(lux)}",
        fontsize=12,
    )

    # --------------------------------------------------------
    # Edge Weight
    # --------------------------------------------------------
    ax = axes[1]
    ax.imshow(
        edge_weight,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.axis("off")
    ax.set_title(
        "Edge Weight",
        fontsize=12,
    )

    # --------------------------------------------------------
    # Final Output
    # --------------------------------------------------------
    ax = axes[2]
    ax.imshow(
        final_vis,
        interpolation="nearest",
    )
    ax.axis("off")
    ax.set_title(
        "Final Output",
        fontsize=12,
    )

    plt.tight_layout(pad=0.35)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

def save_pipeline_comparison_plot(
    original_img,
    records,
    save_path,
    mode="y_only",
    hist_bins=32,
):
    """
    生成两阶段处理的对比图。

    records: list[dict]
        每个元素包含：
        {
            "lux": lux,
            "p": p,
            "s": s,
            "stage1_img": stage1_img,
            "stage2_img": stage2_img,
            "edge_params": edge_params,
        }
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_levels = len(records)
    n_cols = 1 + n_levels

    fig_w = max(15, 4.0 * n_cols)
    fig_h = 9.5

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)

    gs = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=[1.0, 1.0, 1.35],
        hspace=0.35,
        wspace=0.14,
    )

    # ========================================================
    # 第一行：Original + Stage1 tone
    # ========================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_to_rgb_float_for_plot(original_img))
    ax.set_title("Original", fontsize=10)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        p = rec["p"]
        s = rec["s"]
        img = rec["stage1_img"]

        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(image_to_rgb_float_for_plot(img))
        ax.set_title(
            f"Stage1 Tone\n{lux_to_name(lux)}, p={p:.2f}, s={s:.2f}",
            fontsize=10,
        )
        ax.axis("off")

    # ========================================================
    # 第二行：Original + Stage2 Sobel edge
    # ========================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(image_to_rgb_float_for_plot(original_img))
    ax.set_title("Original Ref", fontsize=10)
    ax.axis("off")

    for i, rec in enumerate(records):
        lux = rec["lux"]
        img = rec["stage2_img"]
        edge_params = rec["edge_params"]

        amount = edge_params.get("amount", 0.0)
        sigma = edge_params.get("sigma", 0.0)

        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(image_to_rgb_float_for_plot(img))
        ax.set_title(
            f"Stage2 Sobel Edge\n{lux_to_name(lux)}, amount={amount:.2f}, sigma={sigma:.2f}",
            fontsize=10,
        )
        ax.axis("off")

    # ========================================================
    # 第三行左侧：原图亮度直方图
    # ========================================================
    ax_hist = fig.add_subplot(gs[2, 0])

    Y = compute_luma_float(original_img)
    hist, edges = np.histogram(Y.reshape(-1), bins=hist_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist = hist / max(np.sum(hist), 1.0)

    centers = (edges[:-1] + edges[1:]) * 0.5 * 255.0
    bin_width = 255.0 / hist_bins * 0.85

    ax_hist.bar(
        centers,
        hist,
        width=bin_width,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.4,
    )

    ax_hist.set_title(f"{hist_bins}-bin Original Luma Hist", fontsize=10)
    ax_hist.set_xlim(0, 255)
    ax_hist.set_xlabel("Luma")
    ax_hist.set_ylabel("Ratio")
    ax_hist.grid(True, linestyle="-", alpha=0.35)

    # ========================================================
    # 第三行右侧：Tone LUT 曲线
    # ========================================================
    ax_curve = fig.add_subplot(gs[2, 1:])

    x_8bit = np.arange(256, dtype=np.float32)
    x = x_8bit / 255.0

    ax_curve.plot(
        x_8bit,
        x_8bit,
        linestyle="--",
        linewidth=1.4,
        color="#1f77b4",
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
        p = rec["p"]
        s = rec["s"]

        y = sunlight_curve(x, p=p, s=s)
        y_8bit = np.clip(np.round(y * 255.0), 0, 255)

        color = colors[i % len(colors)]

        ax_curve.plot(
            x_8bit,
            y_8bit,
            linewidth=1.8,
            color=color,
            label=f"{lux_to_name(lux)}, p={p:.2f}, s={s:.2f}",
        )

        ax_curve.scatter(
            [p * 255.0],
            [p * 255.0],
            s=18,
            color=color,
            zorder=5,
        )

    ax_curve.set_title(f"Stage1 Sunlight Tone LUTs, mode={mode}", fontsize=10)
    ax_curve.set_xlim(0, 255)
    ax_curve.set_ylim(0, 255)
    ax_curve.set_xlabel("Input")
    ax_curve.set_ylabel("Output")
    ax_curve.grid(True, linestyle="-", alpha=0.35)
    ax_curve.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 8. 批量处理文件夹
# ============================================================

def is_image_file(path):
    suffix = path.suffix.lower()
    return suffix in [
        ".jpg", ".jpeg",
        ".png",
        ".bmp",
        ".tif", ".tiff",
        ".webp",
    ]


def process_folder(
    input_dir,
    output_dir,
    lux_list,
    mode="y_only",
    recursive=False,
    overwrite=True,
    save_compare=True,
    compare_dir=None,
    hist_bins=32,
    save_edge=True,
    edge_scale=1.0,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if compare_dir is None:
        compare_dir = output_dir / "_compare"
    else:
        compare_dir = Path(compare_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    if recursive:
        image_paths = [
            p for p in input_dir.rglob("*")
            if p.is_file() and is_image_file(p)
        ]
    else:
        image_paths = [
            p for p in input_dir.glob("*")
            if p.is_file() and is_image_file(p)
        ]

    if len(image_paths) == 0:
        print(f"No image found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Stage1 mode: {mode}")
    print(f"Save Stage2 Sobel edge: {save_edge}")
    print(f"Edge amount scale: {edge_scale}")
    print(f"Lux list: {lux_list}")

    print()
    print("Stage1 tone curve parameters:")
    for lux in lux_list:
        p, s = get_params_by_lux(lux)
        print(f"  {lux_to_name(lux)}: p = {p:.4f}, s = {s:.4f}")

    print()
    print("Stage2 Sobel edge parameters:")
    for lux in lux_list:
        edge_params = get_edge_params_by_lux(lux)
        edge_params["amount"] *= edge_scale
        print(
            f"  {lux_to_name(lux)}: "
            f"amount = {edge_params['amount']:.4f}, "
            f"sigma = {edge_params['sigma']:.4f}, "
            f"edge_low = {edge_params['edge_low']:.4f}, "
            f"edge_high = {edge_params['edge_high']:.4f}"
        )

    print()
    print("=" * 70)
    print("Start processing images")
    print("=" * 70)

    for idx, img_path in enumerate(image_paths):
        rel_path = img_path.relative_to(input_dir)

        print()
        print(f"[Image {idx + 1:04d}/{len(image_paths):04d}] {img_path}")

        original_img = imread_unicode(img_path, cv2.IMREAD_UNCHANGED)

        if original_img is None:
            print(f"  [Failed] Read image: {img_path}")
            continue

        records = []

        for lux in lux_list:
            lux_name = lux_to_name(lux)

            # =================================================
            # Stage 1：Y 通道 tone curve 增强
            # =================================================
            p, s = get_params_by_lux(lux)

            stage1_dir = output_dir / "stage1_tone" / lux_name
            stage1_path = stage1_dir / rel_path

            stage1_img = None

            if stage1_path.exists() and not overwrite:
                print(f"  [Skip] Stage1 exists: {stage1_path}")
                stage1_img = imread_unicode(stage1_path, cv2.IMREAD_UNCHANGED)

                if stage1_img is None:
                    print(f"  [Warning] Failed to read existing Stage1: {stage1_path}")
                    continue
            else:
                try:
                    stage1_img = enhance_image(
                        original_img,
                        p=p,
                        s=s,
                        mode=mode,
                    )
                except Exception as e:
                    print(f"  [Failed] Stage1 {lux_name}, error: {e}")
                    continue

                imwrite_unicode(stage1_path, stage1_img)
                print(f"  [Save] Stage1 {lux_name} -> {stage1_path}")

            # =================================================
            # Stage 2：Sobel 边缘增强
            # 基于 Stage1 结果继续处理
            # =================================================
            stage2_img = stage1_img
            edge_params = get_edge_params_by_lux(lux)
            edge_params["amount"] *= edge_scale

            if save_edge:
                stage2_dir = output_dir / "stage2_sobel" / lux_name
                stage2_path = stage2_dir / rel_path

                if stage2_path.exists() and not overwrite:
                    print(f"  [Skip] Stage2 exists: {stage2_path}")
                    stage2_img = imread_unicode(stage2_path, cv2.IMREAD_UNCHANGED)

                    if stage2_img is None:
                        print(f"  [Warning] Failed to read existing Stage2: {stage2_path}")
                        stage2_img = stage1_img
                else:
                    try:
                        stage2_img = sobel_edge_enhance_image(
                            stage1_img,
                            amount=edge_params["amount"],
                            sigma=edge_params["sigma"],
                            edge_low=edge_params["edge_low"],
                            edge_high=edge_params["edge_high"],
                            edge_blur=edge_params["edge_blur"],
                            luma_low=edge_params["luma_low"],
                            luma_high=edge_params["luma_high"],
                            norm_percentile=edge_params["norm_percentile"],
                            standard="bt709",
                        )
                    except Exception as e:
                        print(f"  [Failed] Stage2 {lux_name}, error: {e}")
                        stage2_img = stage1_img
                        continue

                    imwrite_unicode(stage2_path, stage2_img)
                    print(f"  [Save] Stage2 {lux_name} -> {stage2_path}")

            records.append(
                {
                    "lux": lux,
                    "p": p,
                    "s": s,
                    "stage1_img": stage1_img,
                    "stage2_img": stage2_img,
                    "edge_params": edge_params,
                }
            )

        # =====================================================
        # 对比图
        # =====================================================
        # =====================================================
        # 分阶段对比图
        # =====================================================
        if save_compare and len(records) > 0:
            stage1_compare_dir = output_dir / "_compare_stage1"
            stage2_compare_dir = output_dir / "_compare_stage2"

            stage1_compare_path = stage1_compare_dir / rel_path.with_suffix(".stage1_compare.png")
            stage2_compare_path = stage2_compare_dir / rel_path.with_suffix(".stage2_compare.png")

            edge_pipeline_dir = output_dir / "_edge_pipeline"
            edge_pipeline_path = edge_pipeline_dir / rel_path.with_suffix(".edge_pipeline.png")

            # -----------------------------
            # Stage1 compare
            # -----------------------------
            if stage1_compare_path.exists() and not overwrite:
                print(f"  [Skip] Stage1 compare exists: {stage1_compare_path}")
            else:
                try:
                    save_stage1_comparison_plot(
                        original_img=original_img,
                        records=records,
                        save_path=stage1_compare_path,
                        mode=mode,
                        hist_bins=hist_bins,
                    )
                    print(f"  [Save] Stage1 compare -> {stage1_compare_path}")
                except Exception as e:
                    print(
                        f"  [Failed] Save Stage1 compare: "
                        f"{stage1_compare_path}, error: {e}"
                    )

            # -----------------------------
            # Stage2 compare
            # -----------------------------
            if save_edge:
                if stage2_compare_path.exists() and not overwrite:
                    print(f"  [Skip] Stage2 compare exists: {stage2_compare_path}")
                else:
                    try:
                        save_stage2_comparison_plot(
                            original_img=original_img,
                            records=records,
                            save_path=stage2_compare_path,
                            hist_bins=hist_bins,
                        )
                        print(f"  [Save] Stage2 compare -> {stage2_compare_path}")
                    except Exception as e:
                        print(
                            f"  [Failed] Save Stage2 compare: "
                            f"{stage2_compare_path}, error: {e}"
                        )

            # -----------------------------
            # Edge pipeline visualization
            # -----------------------------
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
# ============================================================
# 9. 可选：导出曲线图
# ============================================================

def save_curve_plot(save_path):
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1.0, 1001)

    plt.figure(figsize=(8, 6), dpi=160)

    plt.plot(
        x,
        x,
        "k--",
        linewidth=1.4,
        alpha=0.7,
        label="y = x"
    )

    colors = {
        5000: "#1f77b4",
        25000: "#ff7f0e",
        45000: "#d62728",
    }

    for lux in [5000, 25000, 45000]:
        p, s = get_params_by_lux(lux)
        y = sunlight_curve(x, p=p, s=s)

        plt.plot(
            x,
            y,
            linewidth=2.4,
            color=colors.get(lux, None),
            label=f"{lux} lux, p={p:.2f}, s={s:.2f}"
        )

        plt.scatter([p], [p], s=35, color=colors.get(lux, None), zorder=5)

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.45)
    plt.xlabel("Input luminance x")
    plt.ylabel("Output luminance y")
    plt.title("Sunlight Readability Tone Curves")
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Curve plot saved to: {save_path}")


# ============================================================
# 10. 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch apply sunlight readability S-curve to images."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/mnt/data/wcm/sunlight_display/image_test",
        help="Input image folder."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output_images",
        help="Output image folder."
    )

    parser.add_argument(
        "--lux",
        type=float,
        nargs="+",
        default=[5000, 25000, 45000],
        help="Lux values. Default: 5000 25000 45000"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="y_only",
        choices=["y_only", "yuv_y_only", "luma", "rgb", "per_channel"],
        help=(
            "Enhancement mode. "
            "y_only/yuv_y_only: convert to YCbCr, adjust Y only, keep chroma unchanged. "
            "luma: adjust luminance by RGB gain. "
            "rgb/per_channel: apply curve to each channel independently."
        )
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images recursively."
    )

    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not overwrite existing output images."
    )

    parser.add_argument(
        "--save_curve",
        type=str,
        default="./output_images/curve.png",
        help="Optional path to save curve plot, e.g. ./curve.png"
    )

    parser.add_argument(
        "--no_edge",
        action="store_true",
        help="Disable Stage2 Sobel edge enhancement."
    )

    parser.add_argument(
        "--edge_scale",
        type=float,
        default=1.0,
        help="Global scale for Sobel edge enhancement amount. Default: 1.0"
    )

    parser.add_argument(
        "--no_compare",
        action="store_true",
        help="Do not generate per-image comparison figures."
    )

    parser.add_argument(
        "--compare_dir",
        type=str,
        default=None,
        help="Comparison figure output folder. Default: output/_compare"
    )

    parser.add_argument(
        "--hist_bins",
        type=int,
        default=32,
        help="Histogram bin number for comparison figure. Default: 32"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        lux_list=args.lux,
        mode=args.mode,
        recursive=args.recursive,
        overwrite=not args.no_overwrite,
        save_compare=not args.no_compare,
        compare_dir=args.compare_dir,
        hist_bins=args.hist_bins,
        save_edge=not args.no_edge,
        edge_scale=args.edge_scale,
    )

    if args.save_curve is not None:
        save_curve_plot(args.save_curve)


if __name__ == "__main__":
    main()