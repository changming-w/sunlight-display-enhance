/*
 * board_dual_peak_detect.h — 板端双峰直方图判定
 *
 * 与 Python 参考实现对齐（instrument/tone_curve_dual_peak.py find_dual_peaks）：
 *   1. 256-bin 直方图（bin = v*256/255，v=255 -> 255）
 *   2. 高斯平滑 sigma=3（19-tap 整数权重，edge padding）
 *   3. Otsu 找谷底 bin t
 *   4. 暗块 p80 = p_dark，亮块 p20 = p_bright（累积直方图）
 *   5. 三条件判定 + 不对称度检查
 *
 * 阈值为"保守从严"配置：尽可能不判双峰（宁回退 S 曲线，不误走 W 曲线）。
 * 依据 16 张实测图的余量：
 *   sep     通过最小 0.264 / 回退最大 0.221（0103 类，谷深同时不合格）
 *   mass    通过最小 0.108 / 回退最大 0.095（0109）
 *   谷深比  通过最大 0.237（67，被 asym 回退）/ 回退最小 0.434（0103）
 *   asym    通过最大 0.60 以下 / 回退 07(1.11) 67(1.02)
 */
#ifndef BOARD_DUAL_PEAK_DETECT_H
#define BOARD_DUAL_PEAK_DETECT_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================
 * 判定阈值（严苛版：尽可能不判双峰）
 *   sep        >= 64 bins        (64/256 = 0.25)
 *   mass       >= 15% 每侧
 *   谷深比     <= 1/4   (0.25)   谷底高度 / 矮峰高度
 *   asym       <= 1/2   (0.50)   |d1-d2| / mean(d1,d2)
 * 实测余量：sep 通过最小 0.310/回退最大 0.264；mass 通过最小 0.189/回退
 * 最大 0.134；谷深比通过最大 0.135/回退最小 0.237；07(1.11) 67(1.02) 回退
 * ============================================================ */
#define DP_MIN_PEAK_SEP_BINS    64
#define DP_MIN_PEAK_MASS_PCT    15
#define DP_VALLEY_RATIO_NUM     1    /* 1/4 = 0.25 */
#define DP_VALLEY_RATIO_DEN     4
#define DP_MAX_ASYM_NUM         1    /* 1/2  = 0.50 */
#define DP_MAX_ASYM_DEN         2

/* sigma=3 高斯核 19-tap，四舍五入整数权重。
 * 和为 252（未归一）——Otsu argmax 与谷深比值均与线性缩放无关，
 * 故无需归一到 2^k。 */
static const int dp_gauss19[19] = {
    0, 0, 2, 5, 8, 14, 21, 27, 32, 34, 32, 27, 21, 14, 8, 5, 2, 0, 0
};

typedef struct {
    int      is_bimodal;    /* 三条件判定结果 */
    int      use_dual_peak; /* is_bimodal && asym OK -> 用 v3 W-bank，否则回退 v2 S-bank */
    int      p_dark;        /* bin 索引 [0,255]，暗块 p80 */
    int      valley;        /* bin 索引 [0,255]，Otsu 谷底 */
    int      p_bright;      /* bin 索引 [0,255]，亮块 p20 */
    float    v_norm;        /* (valley+0.5)/256，v3 bank v 轴索引 */
    float    d_mean_norm;   /* d_mean/256，v3 bank d 轴索引 */
    float    asym;          /* |d1-d2|/mean(d1,d2) */
    uint64_t total_px;      /* 参与统计的像素数 */
    uint64_t n_dark;        /* 暗块像素数 */
    uint64_t n_bright;      /* 亮块像素数 */
} dp_result_t;

/*
 * y: 8-bit luma 平面（BT.709）
 * 返回 0 成功（result 已填充，无论判定真假）；<0 参数/数据错误。
 * 纯整数 + 少量 double（Otsu），无动态内存，无依赖。
 */
static int dp_detect(const uint8_t *y, int w, int h, int stride,
                     dp_result_t *out)
{
    if (!y || !out || w <= 0 || h <= 0 || stride < w)
        return -1;

    /* ---------- 1. 直方图 ---------- */
    uint32_t hist[256] = {0};
    for (int r = 0; r < h; r++) {
        const uint8_t *row = y + (size_t)r * (size_t)stride;
        for (int c = 0; c < w; c++) {
            unsigned b = ((unsigned)row[c] * 256u) / 255u;
            if (b > 255u)
                b = 255u;
            hist[b]++;
        }
    }

    uint64_t total = 0;
    for (int i = 0; i < 256; i++)
        total += hist[i];
    if (total == 0)
        return -2;

    out->total_px = total;

    /* ---------- 2. 高斯平滑（19-tap，edge padding，和不必归一）---------- */
    uint32_t hs[256];
    for (int i = 0; i < 256; i++) {
        uint32_t acc = 0;
        for (int j = 0; j < 19; j++) {
            int idx = i + j - 9;
            if (idx < 0)   idx = 0;
            if (idx > 255) idx = 255;
            acc += hist[idx] * (uint32_t)dp_gauss19[j];
        }
        hs[i] = acc;
    }

    /* ---------- 3. Otsu 谷底（在平滑直方图上）---------- */
    {
        double t_total = 0.0, sum_all = 0.0;
        for (int i = 0; i < 256; i++) {
            t_total += (double)hs[i];
            sum_all += (double)i * (double)hs[i];
        }
        double wB = 0.0, sumB = 0.0, best = -1.0;
        int t = 128;
        for (int i = 0; i < 255; i++) {
            wB += (double)hs[i];
            sumB += (double)i * (double)hs[i];
            double wF = t_total - wB;
            if (wB <= 0.0 || wF <= 0.0)
                continue;
            double mB = sumB / wB;
            double mF = (sum_all - sumB) / wF;
            double between = wB * wF * (mB - mF) * (mB - mF);
            if (between > best) {
                best = between;
                t = i;
            }
        }
        out->valley = t;
    }
    const int t = out->valley;

    /* ---------- 4. 两侧像素数（原始直方图）---------- */
    uint64_t n_dark = 0;
    for (int i = 0; i <= t; i++)
        n_dark += hist[i];
    uint64_t n_bright = total - n_dark;
    out->n_dark = n_dark;
    out->n_bright = n_bright;
    if (n_dark == 0 || n_bright == 0) {
        out->is_bimodal = 0;
        out->use_dual_peak = 0;
        return 0;
    }

    /* ---------- 5. 锚点：暗块 p80 / 亮块 p10（向中间收） ---------- */
    /* 累积像素数 * 100 >= n * q 时的最小 bin（对齐 np.percentile 语义的整数近似）*/
    int p_dark = t;
    {
        uint64_t cum = 0;
        for (int i = 0; i <= t; i++) {
            cum += hist[i];
            if (cum * 100u >= n_dark * 80u) {
                p_dark = i;
                break;
            }
        }
    }
    int p_bright = 255;
    {
        uint64_t cum = 0;
        for (int i = t + 1; i < 256; i++) {
            cum += hist[i];
            if (cum * 100u >= n_bright * 10u) {
                p_bright = i;
                break;
            }
        }
    }

    /* ---------- 6. 峰高 / 谷深 ---------- */
    uint32_t peak_d = 0, peak_b = 0;
    for (int i = 0; i <= t; i++)
        if (hs[i] > peak_d) peak_d = hs[i];
    for (int i = t + 1; i < 256; i++)
        if (hs[i] > peak_b) peak_b = hs[i];
    uint32_t min_peak = peak_d < peak_b ? peak_d : peak_b;

    /* ---------- 7. 三条件判定（从严）---------- */
    int sep_ok   = (p_bright - p_dark) >= DP_MIN_PEAK_SEP_BINS;
    int mass_ok  = (n_dark   * 100u >= total * (unsigned)DP_MIN_PEAK_MASS_PCT)
                && (n_bright * 100u >= total * (unsigned)DP_MIN_PEAK_MASS_PCT);
    int depth_ok = min_peak == 0
                || (uint64_t)hs[t] * DP_VALLEY_RATIO_DEN
                    <= (uint64_t)min_peak * DP_VALLEY_RATIO_NUM;
    out->is_bimodal = sep_ok && mass_ok && depth_ok;

    /* ---------- 8. 锚点贴边保护（对齐 Python clip）----------
     * p_dark   in [2, t-3]     (0.005 / valley-0.01)
     * p_bright in [t+3, 254]   (valley+0.01 / 0.995)              */
    if (p_dark < 2)            p_dark = 2;
    if (p_dark > t - 3)        p_dark = t - 3;
    if (p_bright < t + 3)      p_bright = t + 3;
    if (p_bright > 254)        p_bright = 254;
    out->p_dark = p_dark;
    out->p_bright = p_bright;

    /* ---------- 9. 不对称度 + 最终分流 ---------- */
    int d1 = t - p_dark;          /* 谷底到暗锚点 */
    int d2 = p_bright - t;        /* 亮锚点到谷底 */
    float asym = (d1 + d2) > 0
               ? (float)(d1 > d2 ? d1 - d2 : d2 - d1) * 2.0f / (float)(d1 + d2)
               : 0.0f;
    out->asym = asym;

    /* asym = 2|d1-d2|/(d1+d2) <= 1/2  <=>  4*|d1-d2| <= d1+d2 */
    int dd = d1 > d2 ? d1 - d2 : d2 - d1;
    int asym_ok = 4 * dd <= d1 + d2;
    out->use_dual_peak = out->is_bimodal && asym_ok;

    /* ---------- 10. bank 索引（归一化，bin 中心）---------- */
    out->v_norm = ((float)t + 0.5f) / 256.0f;
    out->d_mean_norm = (float)(d1 + d2) * 0.5f / 256.0f;

    return 0;
}

#endif /* BOARD_DUAL_PEAK_DETECT_H */
