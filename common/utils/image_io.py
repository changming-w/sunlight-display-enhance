# image_io.py

from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg",
    ".png",
    ".bmp",
    ".tif", ".tiff",
    ".webp",
}


def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """
    支持中文路径的图像读取。
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None

    if data.size == 0:
        return None

    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path, img):
    """
    支持中文路径的图像保存。
    """
    if img is None:
        raise ValueError("Input image is None.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
        path = path.with_suffix(ext)

    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def collect_image_paths(input_dir, recursive=False):
    """
    收集输入目录下的图片路径。
    """
    input_dir = Path(input_dir)

    if recursive:
        paths = [
            p for p in input_dir.rglob("*")
            if p.is_file() and is_image_file(p)
        ]
    else:
        paths = [
            p for p in input_dir.glob("*")
            if p.is_file() and is_image_file(p)
        ]

    return sorted(paths, key=lambda p: str(p))