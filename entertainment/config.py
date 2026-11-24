# entertainment_config.py
#
# 娱乐屏增强曲线配置
# 曲线公式：L_out = (log(1 + k*L_in) / log(1 + k)) * (1 - a) + a
#
# k: 对数映射强度，k 越大高亮压缩越强，暗部提升越明显
# a: 环境光偏移（ambient offset），表示强光下屏幕黑位的最低可见亮度

ENTERTAINMENT_LUX_PRESETS = {
    5000: {
        "k": 1.0,
        "a": 0.02,
    },
    45000: {
        "k": 2.5,
        "a": 0.08,
    },
    100000: {
        "k": 3.8,
        "a": 0.12,
    },
}

ENTERTAINMENT_DEFAULT_LUX_LIST = [5000, 45000, 100000]


# ============================================================
# 娱乐屏边缘增强参数（纯锐化，无描边）
#
# 与仪表屏区别：
#   - outline_strength = 0（不做高反差描边）
#   - amount 更温和（视频/游戏画面不需要过强锐化）
#   - sigma 较小（细节锐化而非粗边轮廓）
#   - edge 阈值适中（避免噪声放大）
# ============================================================

ENTERTAINMENT_EDGE_PRESETS = {
    5000: {
        "amount": 0.80,
        "sigma": 0.80,
        "outline_strength": 0.0,
        "edge_low": 0.06,
        "edge_high": 0.25,
        "edge_blur": 0.90,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
    },
    45000: {
        "amount": 1.50,
        "sigma": 1.00,
        "outline_strength": 0.0,
        "edge_low": 0.04,
        "edge_high": 0.18,
        "edge_blur": 0.70,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
    },
    100000: {
        "amount": 2.20,
        "sigma": 1.20,
        "outline_strength": 0.0,
        "edge_low": 0.03,
        "edge_high": 0.14,
        "edge_blur": 0.60,
        "luma_low": 0.02,
        "luma_high": 0.10,
        "norm_percentile": 99.5,
    },
}


# ============================================================
# 娱乐屏色度补偿参数
#
# 在 Y-only 对数曲线提亮后，视觉上会出现饱和度下降的现象。
# 通过对 Cb/Cr 做与 Y 提升比率相关的等比例缩放，部分恢复饱和度。
#
# 与仪表屏区别：
#   - 娱乐屏整体提亮幅度更大（对数曲线对暗部抬升更猛），
#     需要更强的补偿强度来抵消视觉饱和度损失。
#   - min_chroma_gain = 1.0：仅对提亮像素 (ratio > 1) 做补偿，
#     压暗像素保持原色度不变。
# ============================================================

ENTERTAINMENT_CHROMA_COMPENSATION = {
    5000: {
        "enabled": True,
        "chroma_preserve": 0.15,    # 低 lux，对数曲线 k=1.0 提亮温和，轻度补偿即可
        "max_chroma_gain": 1.3,
        "min_chroma_gain": 1.0,
    },
    45000: {
        "enabled": True,
        "chroma_preserve": 0.30,    # 中 lux，k=2.5 提亮较明显，中等补偿
        "max_chroma_gain": 1.5,
        "min_chroma_gain": 1.0,
    },
    100000: {
        "enabled": True,
        "chroma_preserve": 0.45,    # 高 lux，k=3.8 提亮强烈，需要更强色度补偿
        "max_chroma_gain": 1.8,
        "min_chroma_gain": 1.0,
    },
}
