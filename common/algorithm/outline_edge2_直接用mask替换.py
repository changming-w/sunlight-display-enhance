import cv2
import numpy as np
import os
import glob


def compute_edge(img, denoise=True):
    """返回连续的边缘强度图 edge (float [0,1]) 和去噪后的引导图。"""
    if denoise:
        tmp = (img * 255).astype(np.uint8)
        tmp = cv2.bilateralFilter(tmp, 5, 40, 5)
        guide = tmp.astype(np.float32) / 255.0
    else:
        guide = img

    gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    edge = np.power(mag, 0.7)
    edge[edge < 0.12] = 0
    return edge


def edge_aware_stroke(img, mode="dark", strength=0.7, edge_width=1,
                      denoise=True, bin_thresh=0.25, color=(0, 0, 0)):
    """
    mode:
      dark   - 软衰减压暗（原方案）
      glow   - 发光
      hard   - 硬替换：边缘位置直接写成纯色 color（默认黑）
      both   - dark+glow
    bin_thresh - hard 模式下，edge 超过该值的像素被写死
    """
    edge = compute_edge(img, denoise=denoise)

    if edge_width > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (edge_width, edge_width))
        edge = cv2.dilate(edge, k)

    result = img.copy()

    if mode == "hard":
        # --- 硬替换：二值化出 mask，直接覆盖为纯色 ---
        mask = (edge >= bin_thresh).astype(np.uint8)
        # 轻微清理：去掉孤立单像素噪点
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        col = np.array(color, dtype=np.float32) / 255.0
        result[mask > 0] = col
        edge_dbg = (mask * 255).astype(np.uint8)
    else:
        # --- 软处理：羽化后按强度叠加 ---
        edge_s = cv2.GaussianBlur(edge, (0, 0), 0.8)
        edge_s = np.clip(edge_s * strength, 0, 1)
        e3 = edge_s[:, :, None]
        if mode in ("dark", "both"):
            result = result * (1 - e3)
        if mode in ("glow", "both"):
            result = result + e3 * (1 - result) * 0.8
        edge_dbg = (edge_s * 255).astype(np.uint8)

    out = np.clip(result * 255, 0, 255).astype(np.uint8)
    return out, edge_dbg


def _make_compare(images, labels, pad=8, bg=(40, 40, 40)):
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    bar = 26
    panels = []
    for im, lb in zip(images, labels):
        if im.shape[0] != h or im.shape[1] != w:
            im = cv2.resize(im, (w, h))
        panel = np.full((h + bar, w, 3), bg, dtype=np.uint8)
        panel[bar:bar + h] = im
        cv2.putText(panel, lb, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
        panels.append(np.full((h + bar, pad, 3), bg, dtype=np.uint8))
    return np.hstack(panels[:-1])


def process_folder(input_dir, output_root, styles,
                   exts=("png", "jpg", "jpeg", "bmp", "webp")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(input_dir, f"*.{e}"))
        files += glob.glob(os.path.join(input_dir, f"*.{e.upper()}"))
    files = sorted(set(files))
    if not files:
        print(f"[warn] 没找到图片: {input_dir}")
        return

    for s in styles:
        os.makedirs(os.path.join(output_root, s["name"]), exist_ok=True)
    debug_dir = os.path.join(output_root, "debug")
    compare_dir = os.path.join(output_root, "compare")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    print(f"共 {len(files)} 张图，{len(styles)} 种风格")
    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        stem = os.path.splitext(fname)[0]
        raw = cv2.imread(fp, cv2.IMREAD_COLOR)
        if raw is None:
            print(f"[skip] {fp}")
            continue
        img = raw.astype(np.float32) / 255.0

        panels, labels = [raw], ["original"]
        for s in styles:
            out, dbg = edge_aware_stroke(
                img,
                mode=s.get("mode", "dark"),
                strength=s.get("strength", 0.7),
                edge_width=s.get("edge_width", 1),
                denoise=s.get("denoise", True),
                bin_thresh=s.get("bin_thresh", 0.25),
                color=s.get("color", (0, 0, 0)),
            )
            cv2.imwrite(os.path.join(output_root, s["name"], fname), out)
            cv2.imwrite(
                os.path.join(debug_dir, f"{stem}_edge_{s['name']}.png"), dbg)
            panels.append(out)
            labels.append(s["name"])

        cv2.imwrite(os.path.join(compare_dir, fname),
                    _make_compare(panels, labels))
        print(f"[{i}/{len(files)}] {fname} done")


if __name__ == "__main__":
    input_dir = "/mnt/data/wcm/sunlight_enhance/test/input1"
    output_root = "/mnt/data/wcm/sunlight_enhance/test/output_stroke2"

    styles = [
        {"name": "dark_s07", "mode": "dark", "strength": 0.7},
        {"name": "hard_t25", "mode": "hard", "bin_thresh": 0.25, "edge_width": 1},
        {"name": "hard_t15", "mode": "hard", "bin_thresh": 0.15, "edge_width": 1},
        {"name": "hard_w2",  "mode": "hard", "bin_thresh": 0.25, "edge_width": 2},
        {"name": "glow",     "mode": "glow", "strength": 0.6},
    ]

    process_folder(input_dir, output_root, styles)