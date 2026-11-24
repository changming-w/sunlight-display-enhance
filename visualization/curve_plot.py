# visualization/curve_plot.py
# Tone curve visualization

from pathlib import Path

import numpy as np

from instrument.config import DEFAULT_LUX_LIST, get_tone_params_by_lux, lux_to_name
from common.algorithm.curve_math import sunlight_curve


def save_curve_plot(save_path, lux_list=None):
    """
    保存 Stage1 tone curve 基础曲线图。
    注意：这里画的是 lux preset 基础曲线，不包含内容自适应后的图像级修正。
    """
    import matplotlib.pyplot as plt

    if lux_list is None:
        lux_list = DEFAULT_LUX_LIST

    x = np.linspace(0.0, 1.0, 1001)

    plt.figure(figsize=(8, 6), dpi=160)

    plt.plot(
        x,
        x,
        "k--",
        linewidth=1.4,
        alpha=0.7,
        label="y = x",
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
    ]

    for i, lux in enumerate(lux_list):
        p, s = get_tone_params_by_lux(lux)
        y = sunlight_curve(x, p=p, s=s)

        color = colors[i % len(colors)]

        plt.plot(
            x,
            y,
            linewidth=2.4,
            color=color,
            label=f"{lux_to_name(lux)}, p={p:.2f}, s={s:.2f}",
        )

        plt.scatter(
            [p],
            [p],
            s=35,
            color=color,
            zorder=5,
        )

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
