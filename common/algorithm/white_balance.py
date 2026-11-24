# white_balance.py

import numpy as np

from common.utils.image_utils import get_dtype_max, restore_dtype


# ============================================================
# sRGB / Rec.709 D65 linear RGB <-> XYZ
# ============================================================

SRGB_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

SRGB_XYZ_TO_RGB = np.array(
    [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ],
    dtype=np.float64,
)


# ============================================================
# CAT02
# ============================================================

CAT02 = np.array(
    [
        [ 0.7328,  0.4296, -0.1624],
        [-0.7036,  1.6975,  0.0061],
        [ 0.0030,  0.0136,  0.9834],
    ],
    dtype=np.float64,
)

CAT02_INV = np.linalg.inv(CAT02)


def xy_to_XYZ(xy):
    """
    xy 白点转 XYZ，默认 Y=1。
    """
    x, y = float(xy[0]), float(xy[1])

    if y <= 1e-8:
        raise ValueError(f"Invalid white xy: {xy}")

    X = x / y
    Y = 1.0
    Z = (1.0 - x - y) / y

    return np.array([X, Y, Z], dtype=np.float64)


def build_cat02_matrix(src_white_xy, dst_white_xy):
    """
    构建 CAT02 XYZ -> XYZ 白点适应矩阵。
    """
    src_XYZ = xy_to_XYZ(src_white_xy)
    dst_XYZ = xy_to_XYZ(dst_white_xy)

    src_cone = CAT02 @ src_XYZ
    dst_cone = CAT02 @ dst_XYZ

    scale = dst_cone / np.maximum(src_cone, 1e-8)
    D = np.diag(scale)

    M = CAT02_INV @ D @ CAT02
    return M.astype(np.float64)


def code_to_linear(x, transfer="gamma", gamma=2.2):
    """
    code RGB -> linear RGB
    """
    x = np.clip(x.astype(np.float32), 0.0, 1.0)

    if transfer == "linear":
        return x

    if transfer == "gamma":
        return np.power(x, float(gamma)).astype(np.float32)

    if transfer == "srgb":
        return np.where(
            x <= 0.04045,
            x / 12.92,
            np.power((x + 0.055) / 1.055, 2.4),
        ).astype(np.float32)

    raise ValueError(f"Unsupported transfer: {transfer}")


def linear_to_code(x, transfer="gamma", gamma=2.2):
    """
    linear RGB -> code RGB
    """
    x = np.clip(x.astype(np.float32), 0.0, 1.0)

    if transfer == "linear":
        return x

    if transfer == "gamma":
        return np.power(x, 1.0 / float(gamma)).astype(np.float32)

    if transfer == "srgb":
        return np.where(
            x <= 0.0031308,
            12.92 * x,
            1.055 * np.power(x, 1.0 / 2.4) - 0.055,
        ).astype(np.float32)

    raise ValueError(f"Unsupported transfer: {transfer}")


class WhiteBalanceShifter:
    def __init__(
        self,
        src_white_xy=(0.3127, 0.3290),
        dst_white_xy=(0.2990, 0.3149),
        preserve_y=True,
    ):
        """
        Stage3 白平衡漂移。

        默认假设：
            输入图像已经在前面完成校准；
            当前白点是 src_white_xy，通常是 D65；
            本模块只做风格白点漂移。
        """
        self.src_white_xy = tuple(src_white_xy)
        self.dst_white_xy = tuple(dst_white_xy)
        self.preserve_y = bool(preserve_y)

        self.cat_matrix = build_cat02_matrix(
            src_white_xy=self.src_white_xy,
            dst_white_xy=self.dst_white_xy,
        )

        self.rgb_to_xyz = SRGB_RGB_TO_XYZ
        self.xyz_to_rgb = SRGB_XYZ_TO_RGB

    def apply_linear_rgb(self, rgb_linear, clip=True, return_stats=False):
        """
        对 RGB 排列的 linear RGB 做白点漂移。
        支持 shape (..., 3)。
        """
        rgb = np.asarray(rgb_linear, dtype=np.float64)

        xyz = rgb @ self.rgb_to_xyz.T
        Y_ori = xyz[..., 1:2]

        xyz_shift = xyz @ self.cat_matrix.T

        if self.preserve_y:
            scale = Y_ori / np.maximum(xyz_shift[..., 1:2], 1e-8)
            xyz_shift = xyz_shift * scale

        rgb_shift = xyz_shift @ self.xyz_to_rgb.T

        oob_channel_mask = (rgb_shift < 0.0) | (rgb_shift > 1.0)
        oob_pixel_mask = np.any(oob_channel_mask, axis=-1)

        stats = {
            "out_of_gamut_ratio": float(np.mean(oob_pixel_mask)),
            "out_of_gamut_channel_ratio": float(np.mean(oob_channel_mask)),
            "min_linear": float(np.min(rgb_shift)),
            "max_linear": float(np.max(rgb_shift)),
        }

        if clip:
            rgb_shift = np.clip(rgb_shift, 0.0, 1.0)

        rgb_shift = rgb_shift.astype(np.float32)

        if return_stats:
            return rgb_shift, stats

        return rgb_shift

    def apply_bgr_image(
        self,
        img,
        transfer="gamma",
        gamma=2.2,
        return_stats=False,
    ):
        """
        对 OpenCV BGR / BGRA 图像做白平衡漂移。

        输入：
            uint8 / uint16 / float
            BGR / BGRA

        输出：
            dtype 与输入一致。
        """
        if img is None:
            return None

        if img.ndim == 2:
            out = img.copy()
            stats = {
                "out_of_gamut_ratio": 0.0,
                "out_of_gamut_channel_ratio": 0.0,
                "min_linear": 0.0,
                "max_linear": 1.0,
            }
            if return_stats:
                return out, stats
            return out

        if img.ndim != 3:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        if img.shape[2] == 3:
            bgr = img
            alpha = None
        elif img.shape[2] == 4:
            bgr = img[..., :3]
            alpha = img[..., 3:4]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        dtype = img.dtype
        max_value = get_dtype_max(img)

        bgr_code = bgr.astype(np.float32) / max_value
        bgr_code = np.clip(bgr_code, 0.0, 1.0)

        rgb_code = bgr_code[..., ::-1]

        rgb_linear = code_to_linear(
            rgb_code,
            transfer=transfer,
            gamma=gamma,
        )

        rgb_shift_linear, stats = self.apply_linear_rgb(
            rgb_linear,
            clip=True,
            return_stats=True,
        )

        rgb_shift_code = linear_to_code(
            rgb_shift_linear,
            transfer=transfer,
            gamma=gamma,
        )

        bgr_shift_code = rgb_shift_code[..., ::-1]

        bgr_out = restore_dtype(bgr_shift_code, dtype)

        if alpha is not None:
            out = np.concatenate([bgr_out, alpha], axis=-1)
        else:
            out = bgr_out

        if return_stats:
            return out, stats

        return out