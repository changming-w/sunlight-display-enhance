/*
 * board_dual_peak_quantile.h — 纯百分位双峰判定（板端轻量版，严苛）
 *
 * 算法（只保留"直方图百分位"一种原语，无平滑、无 Otsu、无峰搜索）：
 *   1. 256-bin 直方图（Y 直接做索引）
 *   2. 一次累积扫描取 11 个等距分位点 p5, p14, ..., p95（每段 9% 像素）
 *   3. 段宽 w[k] = qs[k+1] - qs[k]（下限 1 bin）
 *      谷段 k = 中间段(2..7)中最宽者；两侧峰 = 各自最窄段中心
 *   4. d1/d2：以谷段中心 V 分块，左块 p80 / 右块 p10（两次累积扫描）
 *   5. 判定四条件（整数）：
 *        密比   w[k] >= 5 * max(w[峰左], w[峰右])   （谷密度 <= 峰密度/5）
 *        峰距   pk_b - pk_d >= 32 bins              （0.125）
 *        asym   |d1-d2|/mean(d1,d2) <= 0.50          （20*|dd| <= 5*(d1+d2)）
 *              V 通道（maxRGB）口径下亮峰拖尾比 Y 更宽，车机 UI 的
 *              asym≈0.44，故从 0.30 放宽到 0.50（Y 口径车机 0.25 不受影响）
 *        峰半宽 min(d1,d2) >= 75 bins               （d_norm >= 0.294：两峰离谷都远，
 *              深黑底+亮内容的车机 UI 特征；浅色/中亮 UI 峰贴近谷，回退 v2。
 *              V 口径车机 minD=75，从 77 放宽 2 bins；全组次高 56，区分度足够）
 *
 * 输出（两支路查表参数，直方图只建一次）：
 *   is_bimodal=1 -> v3 双峰 bank：
 *        v_norm = (V+0.5)/256   （bank v 轴查询）
 *        d_norm = min(d1,d2)/255（bank d 轴查询，对称锚点 v±min 向谷收）
 *   is_bimodal=0 -> v2 S 曲线 bank：
 *        p50   = p50_bin/255                （bank p50 轴查询）
 *        p_iqr = (p25 + p75) / 2 / 255      （bank p_iqr 轴查询；
 *                与 scripts/apply_lut_bank.py 口径一致）
 *   p25/p50/p75 与分位点同整数语义：cum*100 >= total*q 的最小 bin
 *
 * 对拍口径（23 张测试图，密比5/峰距32/asym0.50/峰半宽≥75bins）：
 * 双峰 1 张（车机：深黑底仪表盘 UI），其余回退 v2；Y 与 V（maxRGB）两口径
 * 均只放行车机（Y: minD=85/asym=0.25；V: minD=75/asym=0.44）。v/d 输出计算
 * 不受阈值影响，C/Python 逐位一致。
 * 注：output/board_lut_v3_input1_6lux_asym030 的 csv 是旧 Otsu+平均 d 口径产物，
 * 与本纯百分位判定不同源，勿作对拍基准。
 *
 * 成本：直方图 O(N) + 3 次累积扫描 O(256) + 整数比较。无浮点必需。
 */
#ifndef BOARD_DUAL_PEAK_QUANTILE_H
#define BOARD_DUAL_PEAK_QUANTILE_H

#include <stdint.h>

#define DQ_SEGMENTS      10    /* 等容段数 K */
#define DQ_FRAC_STEP_PCT 9     /* 每段像素占比 9% */
#define DQ_RATIO         5     /* 峰/谷密度比阈值：w[k] >= 5 * max(w_minL, w_minR) */
#define DQ_MIN_DIST_BINS 32    /* 峰距 >= 32 bins (0.125) */
#define DQ_MAX_ASYM_NUM  5     /* asym <= 5/10 = 0.50（V 口径车机 0.44 放行，
                                  Y 口径车机 0.25 亦过；其余图被峰半宽拦下） */
#define DQ_MAX_ASYM_DEN  10
#define DQ_MIN_D_BINS    75    /* 峰半宽 min(d1,d2) >= 75 bins（d_norm>=0.294），
                                  只放行深黑底+亮内容的车机 UI 型双峰 */

typedef struct {
    int   is_bimodal;    /* 四条件全过 */
    int   k_valley;      /* 谷段索引 [2,7] */
    int   v_bin;         /* 谷段中心（取整 bin） */
    float v_norm;        /* (v_bin+0.5)/256，[0,1] */
    float d_norm;        /* min(d1,d2)/255，[0,1]（对称锚点 v±d 向谷收） */
    int   d1, d2;        /* 左块 p80 距 / 右块 p10 距（bins） */
    float asym;          /* |d1-d2|/mean */
    float p_dark;        /* 左峰位（最窄段中心），[0,1] */
    float p_bright;      /* 右峰位，[0,1] */
    float p50;           /* v2 支路：中位 bin/255，[0,1]（bank p50 轴查询） */
    float p_iqr;         /* v2 支路：(p25+p75)/2/255，[0,1]（bank p_iqr 轴查询） */
} dq_result_t;

/*
 * y: Y 平面（8bit），w/h: 尺寸，stride: 行跨度（字节）
 * 返回 0 成功；out 填充判定结果。
 */
static int dq_detect(const uint8_t *y, int w, int h, int stride,
                     dq_result_t *out)
{
    uint32_t hist[256];
    uint64_t total = (uint64_t)w * (uint64_t)h;
    int i, x, yy;

    out->is_bimodal = 0;
    out->k_valley = -1;
    out->v_bin = 128;
    out->v_norm = 0.5f;
    out->d_norm = 0.0f;
    out->d1 = out->d2 = 0;
    out->asym = 0.0f;
    out->p_dark = 0.0f;
    out->p_bright = 1.0f;
    out->p50 = 0.5f;
    out->p_iqr = 0.5f;
    if (!y || !out || w <= 0 || h <= 0)
        return -1;

    /* ---------- 1. 直方图 ---------- */
    for (i = 0; i < 256; i++) hist[i] = 0;
    for (yy = 0; yy < h; yy++) {
        const uint8_t *row = y + (size_t)yy * stride;
        for (x = 0; x < w; x++)
            hist[row[x]]++;
    }

    /* ---------- 2. 一次累积扫描取 11 个分位点 + p25/p50/p75 ----------
     * q[j] = 5 + 9j (%)，j = 0..10；整数语义：
     * qs[j] = 满足 cum*100 >= total*q[j] 的最小 bin。
     * v2 支路的 p25/p50/p75 在同一循环顺带捕获。
     */
    {
        int qs[DQ_SEGMENTS + 1];
        int p25_bin = 255, p50_bin = 255, p75_bin = 255;  /* 兜底（平图） */
        int j = 0, got25 = 0, got50 = 0, got75 = 0;
        uint64_t cum = 0;
        for (i = 0; i < 256; i++) {
            cum += hist[i];
            while (j <= DQ_SEGMENTS &&
                   cum * 100u >= total * (uint64_t)(5 + DQ_FRAC_STEP_PCT * j)) {
                qs[j++] = i;
            }
            if (!got25 && cum * 100u >= total * 25u) { p25_bin = i; got25 = 1; }
            if (!got50 && cum * 100u >= total * 50u) { p50_bin = i; got50 = 1; }
            if (!got75 && cum * 100u >= total * 75u) { p75_bin = i; got75 = 1; }
            if (j > DQ_SEGMENTS && got75)
                break;   /* 分位点与 p25/p50/p75 均已到手 */
        }
        while (j <= DQ_SEGMENTS) qs[j++] = 255;  /* 兜底（平图） */
        /* v2 支路输出：p50 / p_iqr=(p25+p75)/2；半和用浮点防整数截断 */
        out->p50   = (float)p50_bin / 255.0f;
        out->p_iqr = (float)(p25_bin + p75_bin) * 0.5f / 255.0f;

        /* ---------- 3. 段宽 / 谷 / 峰 ---------- */
        {
            int w_seg[DQ_SEGMENTS];
            int k, i_d, i_b, wk, wd, wb, dist2;

            for (k = 0; k < DQ_SEGMENTS; k++) {
                w_seg[k] = qs[k + 1] - qs[k];
                if (w_seg[k] < 1) w_seg[k] = 1;   /* 下限 1 bin */
            }

            /* 谷段 = 中间 [2,7] 中最宽 */
            k = 2;
            for (i = 3; i <= DQ_SEGMENTS - 3; i++)
                if (w_seg[i] > w_seg[k]) k = i;

            /* 两侧峰 = 各自最窄段 */
            i_d = 0;
            for (i = 1; i < k; i++)
                if (w_seg[i] < w_seg[i_d]) i_d = i;
            i_b = k + 1;
            for (i = k + 2; i < DQ_SEGMENTS; i++)
                if (w_seg[i] < w_seg[i_b]) i_b = i;

            wk = w_seg[k];
            wd = w_seg[i_d];
            wb = w_seg[i_b];

            out->k_valley = k;
            out->p_dark   = (float)(qs[i_d] + qs[i_d + 1]) * 0.5f / 255.0f;
            out->p_bright = (float)(qs[i_b] + qs[i_b + 1]) * 0.5f / 255.0f;

            /* 峰距（bin）：两侧峰位中心差 */
            dist2 = (qs[i_b] + qs[i_b + 1]) - (qs[i_d] + qs[i_d + 1]);

            /* ---------- 4. 谷段中心 V + 分块 p80/p10 ---------- */
            {
                int V = (qs[k] + qs[k + 1] + 1) / 2;   /* 四舍五入 */
                uint64_t n_dark = 0, n_bright, c;
                int p80 = V, p10 = 255;

                if (V < 0) V = 0;
                if (V > 255) V = 255;
                for (i = 0; i <= V; i++) n_dark += hist[i];
                n_bright = total - n_dark;

                out->v_bin = V;
                out->v_norm = ((float)V + 0.5f) / 256.0f;

                if (n_dark == 0 || n_bright == 0) {
                    out->is_bimodal = 0;
                    return 0;
                }

                c = 0;
                for (i = 0; i <= V; i++) {
                    c += hist[i];
                    if (c * 100u >= n_dark * 80u) { p80 = i; break; }
                }
                c = 0;
                for (i = V + 1; i < 256; i++) {
                    c += hist[i];
                    if (c * 100u >= n_bright * 10u) { p10 = i; break; }
                }

                out->d1 = V - p80;
                out->d2 = p10 - V;
                /* d 取两侧较近者（min）：对称锚点 v±d 不超出任一侧真实分位锚点，
                 * 两侧均向谷收（否则暗侧 p80 外扩会连带把亮侧锚点外推贴白边） */
                out->d_norm = (float)(out->d1 < out->d2 ? out->d1 : out->d2)
                            / 255.0f;
                {
                    int dd = out->d1 > out->d2 ? out->d1 - out->d2
                                               : out->d2 - out->d1;
                    int sum = out->d1 + out->d2;
                    /* mean 用浮点半和（sum*0.5f）：
                     * 整数 sum/2 在奇数时截断，且 sum==1 时除零得 inf */
                    out->asym = sum > 0 ? (float)dd / ((float)sum * 0.5f)
                                       : 0.0f;

                    /* ---------- 5. 判定（整数，四条件） ---------- */
                    out->is_bimodal =
                        (wk >= DQ_RATIO * (wd > wb ? wd : wb)) &&
                        (dist2 / 2 >= DQ_MIN_DIST_BINS) &&
                        (2 * DQ_MAX_ASYM_DEN * dd <= DQ_MAX_ASYM_NUM * sum) &&
                        ((out->d1 < out->d2 ? out->d1 : out->d2) >=
                         DQ_MIN_D_BINS);
                }
            }
        }
    }
    return 0;
}

#endif /* BOARD_DUAL_PEAK_QUANTILE_H */
