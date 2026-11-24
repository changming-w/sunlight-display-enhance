import cv2
import numpy as np
import os
import glob


def _compute_luma(base, mode="Y"):
    """
    提取亮度域，两者都是 [0,1]：
      Y : BT.709 luma（0.2126R + 0.7152G + 0.0722B），与
          instrument/tone_curve.py 的 Y 口径一致；
          注意 cv2.COLOR_BGR2GRAY 用的是 BT.601，与这里不同；
      L : CIE Lab L*，感知亮度，暗部对比更敏感，但需要 BGR->LAB 转换。
    """
    mode = str(mode).upper()
    if mode == "L":
        lab = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        return lab[:, :, 0].astype(np.float32) / 255.0
    if mode == "Y":
        # BT.709 系数，直接在 float 域加权，避免 uint8 量化；BGR 顺序
        return (base[..., 0] * 0.0722
                + base[..., 1] * 0.7152
                + base[..., 2] * 0.2126)
    raise ValueError(f"Unknown luma mode: {mode}")


def directional_stroke(img, strength=1.2, radius=1.5, denoise=True,
                       edge_gate=0.06, luma="Y"):
    """
    方向自适应描边（本质=保护性 USM）：
      - 暗于周围的像素 -> 压得更暗（得到暗边）
      - 亮于周围的像素 -> 提得更亮（得到亮边）
    自动适配白字/黑字，不染色、不发糊。
    img:  float32 [0,1] BGR
    luma: 亮度域，"Y"=BT.709 luma（默认，与 tone_curve.py 的 Y 口径一致）
          "L"=CIE Lab L*（旧行为）
    """
    if denoise:
        tmp = (img * 255).astype(np.uint8)
        tmp = cv2.bilateralFilter(tmp, 5, 30, 5)
        base = tmp.astype(np.float32) / 255.0
    else:
        base = img

    # 在亮度域算高频，避免逐通道染色
    lum = _compute_luma(base, luma)

    blur = cv2.GaussianBlur(lum, (0, 0), radius)
    high = lum - blur          # 高频：正=比周围亮，负=比周围暗

    # 门限：弱高频(压缩噪声/平坦区)不动，只增强真实边
    mag = np.abs(high)
    gate = np.clip((mag - edge_gate) / (edge_gate + 1e-6), 0, 1)
    high = high * gate

    lum_new = np.clip(lum + high * strength, 0, 1)

    # 把增强后的亮度差作为比例，等比例施加到 BGR（保持色相，不染色）
    ratio = (lum_new + 1e-6) / (lum + 1e-6)
    out = base * ratio[:, :, None]
    out = np.clip(out, 0, 1)

    dbg = np.clip(np.abs(high) * 4, 0, 1)  # 调试：被增强的边
    return (out * 255).astype(np.uint8), (dbg * 255).astype(np.uint8)


def _make_compare(images, labels, pad=8, bg=(40, 40, 40)):
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    bar = 26
    panels = []
    for im, lb in zip(images, labels):
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        p = np.full((h + bar, w, 3), bg, np.uint8)
        p[bar:] = im
        cv2.putText(p, lb, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels += [p, np.full((h + bar, pad, 3), bg, np.uint8)]
    return np.hstack(panels[:-1])


def process_folder(input_dir, output_root, styles,
                   exts=("png", "jpg", "jpeg", "bmp", "webp")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(input_dir, f"*.{e}"))
        files += glob.glob(os.path.join(input_dir, f"*.{e.upper()}"))
    files = sorted(set(files))
    for s in styles:
        os.makedirs(os.path.join(output_root, s["name"]), exist_ok=True)
    os.makedirs(os.path.join(output_root, "debug"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "compare"), exist_ok=True)

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        stem = os.path.splitext(fname)[0]
        raw = cv2.imread(fp, cv2.IMREAD_COLOR)
        if raw is None:
            continue
        img = raw.astype(np.float32) / 255.0
        panels, labels = [raw], ["original"]
        for s in styles:
            out, dbg = directional_stroke(
                img, strength=s.get("strength", 1.2),
                radius=s.get("radius", 1.5),
                edge_gate=s.get("edge_gate", 0.06),
                denoise=s.get("denoise", True),
                luma=s.get("luma", "Y"))
            cv2.imwrite(os.path.join(output_root, s["name"], fname), out)
            cv2.imwrite(os.path.join(output_root, "debug",
                                     f"{stem}_{s['name']}.png"), dbg)
            panels.append(out)
            labels.append(s["name"])
        cv2.imwrite(os.path.join(output_root, "compare", fname),
                    _make_compare(panels, labels))
        print(f"[{i}/{len(files)}] {fname} done")


if __name__ == "__main__":
    input_dir = "/mnt/data/wcm/sunlight_enhance/test/output28/stage1_tone/80000lux"
    output_root = "/mnt/data/wcm/sunlight_enhance/test/output_stroke_usm"
    styles = [
        {"name": "L1",   "strength": 0.8, "radius": 1.2, "edge_gate": 0.06},
        {"name": "L2",    "strength": 1.2, "radius": 1.6, "edge_gate": 0.06},
        {"name": "L3", "strength": 1.8, "radius": 2.0, "edge_gate": 0.05},
        {"name": "L4",  "strength": 2.0, "radius": 2.2, "edge_gate": 0.05},
    ]
    process_folder(input_dir, output_root, styles)