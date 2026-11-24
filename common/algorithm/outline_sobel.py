import cv2
import numpy as np
import os
import glob


def edge_aware_stroke(img,
                      mode="dark",        # dark 黑描边 / glow 发光 / both
                      strength=0.7,       # 描边强度 0~1
                      edge_width=1,       # 描边粗细
                      denoise=True):      # 抗压缩
    """返回 (结果图, edge调试图)。输入 img 为 float32 [0,1] 的 BGR。"""
    # 抗压缩：双边滤波保边去块效应
    if denoise:
        tmp = (img * 255).astype(np.uint8)
        tmp = cv2.bilateralFilter(tmp, 5, 40, 5)
        guide = tmp.astype(np.float32) / 255.0
    else:
        guide = img

    gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # --- 用梯度幅值作为“边缘强度图”，不做任何前景判断 ---
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)

    # 非线性增强：抑制弱噪声边，突出强边
    edge = np.power(mag, 0.7)
    edge[edge < 0.12] = 0

    if edge_width > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (edge_width, edge_width))
        edge = cv2.dilate(edge, k)

    # edge = cv2.GaussianBlur(edge, (0, 0), 0.1)  # 羽化
    edge = np.clip(edge * strength, 0, 1)
    edge3 = edge[:, :, None]

    result = img.copy()
    if mode in ("dark", "both"):
        result = result * (1 - edge3)
    if mode in ("glow", "both"):
        result = result + edge3 * (1 - result) * 0.8

    out = np.clip(result * 255, 0, 255).astype(np.uint8)
    edge_dbg = (edge * 255).astype(np.uint8)
    return out, edge_dbg


def process_folder(input_dir,
                   output_root,
                   styles,
                   exts=("png", "jpg", "jpeg", "bmp", "webp")):
    """
    批量处理，多风格对比 + debug。
    styles: list of dict，每个元素是一种风格配置，例如：
        {"name": "dark",      "mode": "dark",  "strength": 0.7, "edge_width": 1}
    输出结构：
        output_root/<风格名>/原文件名
        output_root/debug/<原文件名去扩展>_edge_<风格名>.png
    """
    files = []
    for e in exts:
        files += glob.glob(os.path.join(input_dir, f"*.{e}"))
        files += glob.glob(os.path.join(input_dir, f"*.{e.upper()}"))
    files = sorted(set(files))

    if not files:
        print(f"[warn] 目录下没有找到图片: {input_dir}")
        return

    # 创建各风格目录 + debug 目录 + 对比拼图目录
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

        panels = [raw]  # 对比拼图：第一张放原图
        labels = ["original"]

        for s in styles:
            out, edge_dbg = edge_aware_stroke(
                img,
                mode=s.get("mode", "dark"),
                strength=s.get("strength", 0.7),
                edge_width=s.get("edge_width", 1),
                denoise=s.get("denoise", True),
            )
            # 1) 风格结果，保持原文件名
            cv2.imwrite(os.path.join(output_root, s["name"], fname), out)
            # 2) debug 边缘图，按 风格名 区分，互不覆盖
            cv2.imwrite(
                os.path.join(debug_dir, f"{stem}_edge_{s['name']}.png"),
                edge_dbg)

            panels.append(out)
            labels.append(s["name"])

        # 3) 生成横向对比拼图（原图 + 各风格）
        compare = _make_compare(panels, labels)
        cv2.imwrite(os.path.join(compare_dir, fname), compare)

        print(f"[{i}/{len(files)}] {fname} done")


def _make_compare(images, labels, pad=8, bg=(40, 40, 40)):
    """把多张同尺寸图横向拼接，顶部写标签，方便一眼对比。"""
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    bar = 26  # 标签栏高度
    canvas_h = h + bar
    panels = []
    for im, lb in zip(images, labels):
        # 统一尺寸（不同图分辨率一致时不会触发）
        if im.shape[0] != h or im.shape[1] != w:
            im = cv2.resize(im, (w, h))
        panel = np.full((canvas_h, w, 3), bg, dtype=np.uint8)
        panel[bar:bar + h] = im
        cv2.putText(panel, lb, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
        panels.append(np.full((canvas_h, pad, 3), bg, dtype=np.uint8))
    panels = panels[:-1]  # 去掉最后一个间隔
    return np.hstack(panels)


if __name__ == "__main__":
    input_dir = "/mnt/data/wcm/sunlight_enhance/test/output28/stage1_tone/80000lux"
    output_root = "/mnt/data/wcm/sunlight_enhance/test/output_stroke_sobel"

    # 在这里定义要对比的风格，想加就加一行
    styles = [
        {"name": "dark_s05",  "mode": "dark", "strength": 0.5, "edge_width": 1},
        {"name": "dark_s07",  "mode": "dark", "strength": 0.7, "edge_width": 1},
        {"name": "dark_s10w1",   "mode": "dark", "strength": 1.0, "edge_width": 1},
        {"name": "glow",      "mode": "glow", "strength": 0.6, "edge_width": 1},
        {"name": "both",      "mode": "both", "strength": 1.0, "edge_width": 1},
    ]

    process_folder(input_dir, output_root, styles)