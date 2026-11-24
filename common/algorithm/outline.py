import cv2
import numpy as np


def guided_filter(src, guide, radius=4, eps=1e-3):
    """导向滤波：保边去噪，用来压制 JPEG/视频压缩的块效应和振铃。"""
    src = src.astype(np.float32)
    guide = guide.astype(np.float32)
    mean_I = cv2.boxFilter(guide, -1, (radius, radius))
    mean_p = cv2.boxFilter(src, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * guide + mean_b


def preprocess_deblock(img_bgr, radius=3, eps=1e-3):
    """抗压缩预处理：对每个通道做导向滤波，自引导，压块效应/振铃。"""
    out = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        ch = img_bgr[:, :, c].astype(np.float32) / 255.0
        out[:, :, c] = guided_filter(ch, ch, radius=radius, eps=eps)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def get_foreground_mask(img_bgr,
                        block_size=15,
                        C=8,
                        min_area=8):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 两种极性都算出来
    th_dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)   # 选中比局部暗的（深色字）
    th_light = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C)       # 选中比局部亮的（浅色字）

    # 关键：自动选极性，而不是 OR。前景应该是“少数像素”，
    # 谁点亮的像素更少（更像前景而非背景），就用谁。
    ratio_dark = th_dark.mean() / 255.0
    ratio_light = th_light.mean() / 255.0

    cand = []
    if 0.001 < ratio_dark < 0.5:
        cand.append((ratio_dark, th_dark))
    if 0.001 < ratio_light < 0.5:
        cand.append((ratio_light, th_light))

    if not cand:
        # 两种极性都异常（要么全黑要么全白），用 Otsu 兜底
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask.mean() > 127:        # 如果前景被反选，翻过来
            mask = cv2.bitwise_not(mask)
    else:
        # 取“点亮像素更少”的那个极性当前景
        cand.sort(key=lambda x: x[0])
        mask = cand[0][1]

    # 形态学清理 + 去小连通域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # 最终健全性检查：如果还是占了超过一半，说明极性反了，翻转
    if clean.mean() > 127:
        clean = cv2.bitwise_not(clean)

    soft = cv2.GaussianBlur(clean.astype(np.float32), (3, 3), 0) / 255.0
    return np.clip(soft, 0, 1)


def make_stroke_mask(mask, thickness=2, mode="outer"):
    """
    由前景 mask 生成描边区域。
      mode: outer(外描边) / inner(内描边) / center(居中)
    返回 0~1 的描边 alpha。
    """
    bin_mask = (mask > 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * thickness + 1, 2 * thickness + 1))
    if mode == "outer":
        dil = cv2.dilate(bin_mask, k)
        stroke = cv2.subtract(dil, bin_mask)
    elif mode == "inner":
        ero = cv2.erode(bin_mask, k)
        stroke = cv2.subtract(bin_mask, ero)
    else:  # center
        dil = cv2.dilate(bin_mask, k)
        ero = cv2.erode(bin_mask, k)
        stroke = cv2.subtract(dil, ero)
    # 羽化让边过渡自然
    stroke = cv2.GaussianBlur(stroke.astype(np.float32), (3, 3), 0) / 255.0
    return np.clip(stroke, 0, 1)


def make_glow_mask(mask, radius=6, intensity=1.0):
    """发光边：膨胀后高斯模糊，得到柔和光晕 alpha。"""
    bin_mask = (mask > 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dil = cv2.dilate(bin_mask, k, iterations=max(1, radius // 3))
    glow = cv2.GaussianBlur(dil.astype(np.float32), (0, 0), radius) / 255.0
    glow = np.clip(glow * intensity, 0, 1)
    # 扣掉前景本体，只留外圈光晕
    glow[bin_mask > 0] = 0
    return glow


def apply_stroke(img_bgr, alpha, color=(0, 0, 0)):
    """把描边/发光 alpha 以指定颜色叠加到原图上。"""
    out = img_bgr.astype(np.float32)
    color = np.array(color, dtype=np.float32)
    a = alpha[:, :, None]
    out = color * a + out * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def stroke_pipeline(img_path,
                    out_path="result.png",
                    style="hard",          # hard / glow / both
                    stroke_color=(0, 0, 0),
                    glow_color=(255, 255, 255),
                    thickness=2,
                    glow_radius=6,
                    do_deblock=True):
    """完整管线：读图 -> 抗压缩 -> 取mask -> 生成描边 -> 合成 -> 保存。"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    work = preprocess_deblock(img) if do_deblock else img
    mask = get_foreground_mask(work)

    result = img.copy()
    if style in ("glow", "both"):
        glow = make_glow_mask(mask, radius=glow_radius)
        result = apply_stroke(result, glow, color=glow_color)
    if style in ("hard", "both"):
        stroke = make_stroke_mask(mask, thickness=thickness, mode="outer")
        result = apply_stroke(result, stroke, color=stroke_color)

    cv2.imwrite(out_path, result)
    # 同时存 mask 方便你调参排查
    cv2.imwrite("debug_mask.png", (mask * 255).astype(np.uint8))
    print(f"done -> {out_path}, mask -> debug_mask.png")
    return result


if __name__ == "__main__":
    # 黑色硬描边
    stroke_pipeline("/mnt/data/wcm/sunlight_enhance/test/input1/05.png", "out_hard.png",
                    style="hard", stroke_color=(0, 0, 0), thickness=2)

    # 白色外发光
    stroke_pipeline("/mnt/data/wcm/sunlight_enhance/test/input1/05.png", "out_glow.png",
                    style="glow", glow_color=(255, 255, 255), glow_radius=8)

    # 黑边 + 白发光（先发光再压黑边，常见的高可读性组合）
    stroke_pipeline("/mnt/data/wcm/sunlight_enhance/test/input1/05.png", "out_both.png",
                    style="both",
                    stroke_color=(0, 0, 0),
                    glow_color=(255, 255, 255),
                    thickness=2, glow_radius=8)