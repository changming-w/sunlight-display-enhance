import cv2
import numpy as np
import os
import glob


def edge_aware_stroke_alpha_no_erode(
        img,
        mode="adaptive",            # adaptive / black / white
        edge_thresh=0.12,           # 边缘阈值，越小边越多
        stroke_radius=1,            # 描边外扩半径，1=约1px
        stroke_alpha=0.65,          # 描边 alpha，0~1
        denoise=True,               # 双边滤波抗压缩块
        gamma=0.7,                  # 边缘非线性增强
        lum_thresh=0.5,             # 亮暗兜底阈值
        local_sigma=2.0,            # 局部亮度估计尺度
        polarity_margin=0.03,       # 局部亮暗判断容忍区
        norm_percentile=99.5,       # Sobel 幅值归一化分位数
        protect_same_polarity=True, # 避免描进同亮暗属性的轮廓内部
        alpha_from_edge=False,      # True: alpha 随边缘强度变化；False: 固定 alpha
        feather_sigma=0.0           # >0 时对描边 alpha 做轻微羽化
):
    """
    不侵蚀轮廓本体的 alpha 混合描边。

    输入:
        img: float32 [0,1], BGR

    输出:
        result: uint8 BGR
        edge_dbg:
            adaptive:
                红色 = 黑描边 alpha 区域
                蓝色 = 白描边 alpha 区域
                绿色 = 被保护的原始轮廓本体
            black / white:
                红色或蓝色 = 描边 alpha 区域
                绿色 = 被保护的原始轮廓本体

    mode:
        adaptive:
            浅色轮廓周围混黑边
            深色轮廓周围混白边
        black:
            所有轮廓周围混黑边
        white:
            所有轮廓周围混白边
    """

    stroke_alpha = float(np.clip(stroke_alpha, 0.0, 1.0))

    # -----------------------------
    # 1. 准备 guide 图
    # -----------------------------
    raw_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    raw_f = raw_u8.astype(np.float32) / 255.0

    if denoise:
        tmp = cv2.bilateralFilter(raw_u8, 5, 40, 5)
        guide_u8 = tmp
    else:
        guide_u8 = raw_u8

    gray_u8 = cv2.cvtColor(guide_u8, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    result_f = raw_f.copy()

    # -----------------------------
    # 2. Sobel 找轮廓本体 edge_core
    # -----------------------------
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    scale = np.percentile(mag, norm_percentile)
    if scale < 1e-6:
        scale = mag.max() + 1e-6

    mag = np.clip(mag / (scale + 1e-6), 0, 1)
    edge_score = np.power(mag, gamma)

    # 原始轮廓本体，后面不直接改它
    edge_core = edge_score >= edge_thresh

    if not np.any(edge_core):
        edge_dbg = np.zeros_like(raw_u8)
        return raw_u8, edge_dbg

    # -----------------------------
    # 3. 准备膨胀 kernel
    # -----------------------------
    stroke_radius = max(1, int(stroke_radius))
    ksize = 2 * stroke_radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ksize, ksize)
    )

    edge_core_u8 = edge_core.astype(np.uint8) * 255
    edge_dilate = cv2.dilate(edge_core_u8, kernel) > 0

    # 基础描边环：只画膨胀出来的一圈，不画 edge_core 本体
    stroke_ring = edge_dilate & (~edge_core)

    edge_dbg = np.zeros_like(raw_u8)

    # -----------------------------
    # 4. 工具函数：生成 alpha map
    # -----------------------------
    def make_alpha_from_mask(mask, seed_score=None):
        """
        mask: bool 描边区域
        seed_score: 可选，边缘强度图
        """
        if alpha_from_edge and seed_score is not None:
            alpha = seed_score.astype(np.float32) * stroke_alpha
            alpha = alpha * mask.astype(np.float32)
        else:
            alpha = mask.astype(np.float32) * stroke_alpha

        if feather_sigma and feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
            # 避免羽化污染原轮廓本体
            alpha[edge_core] = 0.0

        return np.clip(alpha, 0.0, 1.0)

    # -----------------------------
    # 5. 工具函数：alpha blend
    # -----------------------------
    def blend_with_color(base, alpha, color_bgr):
        """
        base: float32 [0,1] BGR
        alpha: float32 [0,1], HxW
        color_bgr: tuple/list, [0,1]
        """
        color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
        a3 = alpha[:, :, None]
        return base * (1.0 - a3) + color * a3

    # -----------------------------
    # 6. 全部描黑
    # -----------------------------
    if mode == "black":
        if alpha_from_edge:
            seed = cv2.dilate(edge_score.astype(np.float32), kernel)
        else:
            seed = None

        alpha_black = make_alpha_from_mask(stroke_ring, seed)
        result_f = blend_with_color(result_f, alpha_black, (0.0, 0.0, 0.0))

        # 恢复原轮廓本体
        result_f[edge_core] = raw_f[edge_core]

        result_u8 = np.clip(result_f * 255, 0, 255).astype(np.uint8)

        edge_dbg[:, :, 2] = np.clip(alpha_black * 255, 0, 255).astype(np.uint8)
        edge_dbg[edge_core] = (0, 255, 0)

        return result_u8, edge_dbg

    # -----------------------------
    # 7. 全部描白
    # -----------------------------
    if mode == "white":
        if alpha_from_edge:
            seed = cv2.dilate(edge_score.astype(np.float32), kernel)
        else:
            seed = None

        alpha_white = make_alpha_from_mask(stroke_ring, seed)
        result_f = blend_with_color(result_f, alpha_white, (1.0, 1.0, 1.0))

        # 恢复原轮廓本体
        result_f[edge_core] = raw_f[edge_core]

        result_u8 = np.clip(result_f * 255, 0, 255).astype(np.uint8)

        edge_dbg[:, :, 0] = np.clip(alpha_white * 255, 0, 255).astype(np.uint8)
        edge_dbg[edge_core] = (0, 255, 0)

        return result_u8, edge_dbg

    # -----------------------------
    # 8. 自适应黑/白描边
    # -----------------------------
    if mode == "adaptive":
        # 局部亮度
        local_gray = cv2.GaussianBlur(gray, (0, 0), local_sigma)
        delta = gray - local_gray

        # 比周围亮：认为是浅色轮廓
        bright_side = delta > polarity_margin

        # 比周围暗：认为是深色轮廓
        dark_side = delta < -polarity_margin

        # 中性区域兜底：按绝对亮度判断
        neutral = ~(bright_side | dark_side)
        bright_side = bright_side | (neutral & (gray >= lum_thresh))
        dark_side = dark_side | (neutral & (gray < lum_thresh))

        # 轮廓本体分类
        black_core = edge_core & bright_side   # 浅色轮廓 -> 黑边
        white_core = edge_core & dark_side     # 深色轮廓 -> 白边

        black_core_u8 = black_core.astype(np.uint8) * 255
        white_core_u8 = white_core.astype(np.uint8) * 255

        black_dilate = cv2.dilate(black_core_u8, kernel) > 0
        white_dilate = cv2.dilate(white_core_u8, kernel) > 0

        black_stroke = black_dilate & (~edge_core)
        white_stroke = white_dilate & (~edge_core)

        # 避免描进同亮暗属性区域内部
        if protect_same_polarity:
            # 浅色轮廓的黑边，尽量画在非浅色侧
            black_stroke = black_stroke & (~bright_side)
            # 深色轮廓的白边，尽量画在非深色侧
            white_stroke = white_stroke & (~dark_side)

        # 冲突区域：根据当前像素亮度选反差更大的颜色
        conflict = black_stroke & white_stroke
        if np.any(conflict):
            conflict_black = conflict & (gray >= lum_thresh)  # 亮区域用黑边
            conflict_white = conflict & (gray < lum_thresh)   # 暗区域用白边

            black_stroke = (black_stroke & (~conflict)) | conflict_black
            white_stroke = (white_stroke & (~conflict)) | conflict_white

        # alpha map
        if alpha_from_edge:
            black_seed = np.where(black_core, edge_score, 0).astype(np.float32)
            white_seed = np.where(white_core, edge_score, 0).astype(np.float32)

            black_seed_dilate = cv2.dilate(black_seed, kernel)
            white_seed_dilate = cv2.dilate(white_seed, kernel)
        else:
            black_seed_dilate = None
            white_seed_dilate = None

        alpha_black = make_alpha_from_mask(black_stroke, black_seed_dilate)
        alpha_white = make_alpha_from_mask(white_stroke, white_seed_dilate)

        # 再次处理 alpha 冲突，避免黑白叠加变灰
        overlap = (alpha_black > 0) & (alpha_white > 0)
        if np.any(overlap):
            black_keep = overlap & (gray >= lum_thresh)
            white_keep = overlap & (gray < lum_thresh)

            alpha_white[black_keep] = 0.0
            alpha_black[white_keep] = 0.0

        # 先混黑，再混白
        result_f = blend_with_color(result_f, alpha_black, (0.0, 0.0, 0.0))
        result_f = blend_with_color(result_f, alpha_white, (1.0, 1.0, 1.0))

        # 关键：恢复原始轮廓本体，避免侵蚀
        result_f[edge_core] = raw_f[edge_core]

        result_u8 = np.clip(result_f * 255, 0, 255).astype(np.uint8)

        # debug: 红=黑边 alpha，蓝=白边 alpha，绿=保护轮廓
        edge_dbg[:, :, 2] = np.maximum(
            edge_dbg[:, :, 2],
            np.clip(alpha_black * 255, 0, 255).astype(np.uint8)
        )
        edge_dbg[:, :, 0] = np.maximum(
            edge_dbg[:, :, 0],
            np.clip(alpha_white * 255, 0, 255).astype(np.uint8)
        )
        edge_dbg[edge_core] = (0, 255, 0)

        return result_u8, edge_dbg

    raise ValueError(f"未知 mode: {mode}, 可选 adaptive / black / white")


def process_folder(
        input_dir,
        output_root,
        styles,
        exts=("png", "jpg", "jpeg", "bmp", "webp"),
        output_ext=".png"
):
    """
    批量处理，多风格对比 + debug。

    输出结构:
        output_root/<风格名>/<原文件名stem>.png
        output_root/debug/<原文件名stem>_edge_<风格名>.png
        output_root/compare/<原文件名stem>.png
    """

    files = []
    for e in exts:
        files += glob.glob(os.path.join(input_dir, f"*.{e}"))
        files += glob.glob(os.path.join(input_dir, f"*.{e.upper()}"))

    files = sorted(set(files))

    if not files:
        print(f"[warn] 目录下没有找到图片: {input_dir}")
        return

    for s in styles:
        os.makedirs(os.path.join(output_root, s["name"]), exist_ok=True)

    debug_dir = os.path.join(output_root, "debug")
    compare_dir = os.path.join(output_root, "compare")

    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    print(f"共 {len(files)} 张图，{len(styles)} 种风格 -> {output_root}")

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        stem = os.path.splitext(fname)[0]

        raw = cv2.imread(fp, cv2.IMREAD_COLOR)
        if raw is None:
            print(f"[skip] 无法读取: {fp}")
            continue

        img = raw.astype(np.float32) / 255.0

        panels = [raw]
        labels = ["original"]

        out_fname = stem + output_ext

        for s in styles:
            out, edge_dbg = edge_aware_stroke_alpha_no_erode(
                img,
                mode=s.get("mode", "adaptive"),
                edge_thresh=s.get("edge_thresh", 0.12),
                stroke_radius=s.get("stroke_radius", 1),
                stroke_alpha=s.get("stroke_alpha", 0.65),
                denoise=s.get("denoise", True),
                gamma=s.get("gamma", 0.7),
                lum_thresh=s.get("lum_thresh", 0.5),
                local_sigma=s.get("local_sigma", 2.0),
                polarity_margin=s.get("polarity_margin", 0.03),
                norm_percentile=s.get("norm_percentile", 99.5),
                protect_same_polarity=s.get("protect_same_polarity", True),
                alpha_from_edge=s.get("alpha_from_edge", False),
                feather_sigma=s.get("feather_sigma", 0.0)
            )

            # 1. 保存结果
            out_path = os.path.join(output_root, s["name"], out_fname)
            _imwrite_lossless(out_path, out)

            # 2. 保存 debug
            dbg_path = os.path.join(debug_dir, f"{stem}_edge_{s['name']}.png")
            _imwrite_lossless(dbg_path, edge_dbg)

            panels.append(out)
            labels.append(s["name"])

        # 3. 保存横向对比图
        compare = _make_compare(panels, labels)
        compare_path = os.path.join(compare_dir, out_fname)
        _imwrite_lossless(compare_path, compare)

        print(f"[{i}/{len(files)}] {fname} done")


def _imwrite_lossless(path, img):
    """
    尽量无损保存。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif ext in [".jpg", ".jpeg"]:
        # 不推荐 JPG。即使 quality=100，仍可能有压缩污染。
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(path, img)


def _make_compare(images, labels, pad=8, bg=(40, 40, 40)):
    """
    多张图横向拼接，顶部写标签。
    """
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)

    bar = 26
    canvas_h = h + bar
    panels = []

    for im, lb in zip(images, labels):
        if im.shape[0] != h or im.shape[1] != w:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)

        panel = np.full((canvas_h, w, 3), bg, dtype=np.uint8)
        panel[bar:bar + h] = im

        cv2.putText(
            panel,
            lb,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        panels.append(panel)
        panels.append(np.full((canvas_h, pad, 3), bg, dtype=np.uint8))

    panels = panels[:-1]
    return np.hstack(panels)


if __name__ == "__main__":
    input_dir = "/mnt/data/wcm/sunlight_enhance/test/output28/stage1_tone/80000lux"
    output_root = "/mnt/data/wcm/sunlight_enhance/test/output_stroke_alpha_no_erode"

    styles = [
        # 推荐：自适应黑/白 alpha 描边
        {
            "name": "adaptive_a045_r1_t012",
            "mode": "adaptive",
            "edge_thresh": 0.12,
            "stroke_radius": 1,
            "stroke_alpha": 0.45,
            "denoise": True,
            "gamma": 0.7,
            "local_sigma": 2.0,
            "polarity_margin": 0.03,
            "protect_same_polarity": True,
            "alpha_from_edge": False,
            "feather_sigma": 0.0,
        },

        {
            "name": "adaptive_a065_r1_t012",
            "mode": "adaptive",
            "edge_thresh": 0.12,
            "stroke_radius": 1,
            "stroke_alpha": 0.65,
            "denoise": True,
            "gamma": 0.7,
            "local_sigma": 2.0,
            "polarity_margin": 0.03,
            "protect_same_polarity": True,
            "alpha_from_edge": False,
            "feather_sigma": 0.0,
        },

        # 稍粗一点
        {
            "name": "adaptive_a065_r2_t012",
            "mode": "adaptive",
            "edge_thresh": 0.12,
            "stroke_radius": 2,
            "stroke_alpha": 0.65,
            "denoise": True,
            "gamma": 0.7,
            "local_sigma": 2.0,
            "polarity_margin": 0.03,
            "protect_same_polarity": True,
            "alpha_from_edge": False,
            "feather_sigma": 0.0,
        },

        # 更保守，少描背景纹理
        {
            "name": "adaptive_a065_r1_t018",
            "mode": "adaptive",
            "edge_thresh": 0.18,
            "stroke_radius": 1,
            "stroke_alpha": 0.65,
            "denoise": True,
            "gamma": 0.7,
            "local_sigma": 2.0,
            "polarity_margin": 0.03,
            "protect_same_polarity": True,
            "alpha_from_edge": False,
            "feather_sigma": 0.0,
        },

        # alpha 随边缘强度变化，更自然一些
        {
            "name": "adaptive_a075_r1_t012_edgealpha",
            "mode": "adaptive",
            "edge_thresh": 0.12,
            "stroke_radius": 1,
            "stroke_alpha": 0.75,
            "denoise": True,
            "gamma": 0.7,
            "local_sigma": 2.0,
            "polarity_margin": 0.03,
            "protect_same_polarity": True,
            "alpha_from_edge": True,
            "feather_sigma": 0.0,
        },

        # 轻微 halo 感，可能更适合文字 UI
        {
            "name": "adaptive_a060_r1_t012_feather",
            "mode": "adaptive",
            "edge_thresh": 0.12,
            "stroke_radius": 1,
            "stroke_alpha": 0.60,
            "denoise": True,
            "gamma": 0.7,
            "local_sigma": 2.0,
            "polarity_margin": 0.03,
            "protect_same_polarity": True,
            "alpha_from_edge": False,
            "feather_sigma": 0.4,
        },

        # 所有边缘统一黑描边
        {
            "name": "all_black_a065_r1_t012",
            "mode": "black",
            "edge_thresh": 0.12,
            "stroke_radius": 1,
            "stroke_alpha": 0.65,
            "denoise": True,
        },

        # 所有边缘统一白描边
        {
            "name": "all_white_a065_r1_t012",
            "mode": "white",
            "edge_thresh": 0.12,
            "stroke_radius": 1,
            "stroke_alpha": 0.65,
            "denoise": True,
        },
    ]

    process_folder(
        input_dir=input_dir,
        output_root=output_root,
        styles=styles,
        output_ext=".png"
    )