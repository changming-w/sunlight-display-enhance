# 环境光 Viewing 仿真方案详解

## 目录

- [一、问题背景](#一问题背景)
- [二、核心物理模型](#二核心物理模型)
  - [2.1 显示屏亮度组成](#21-显示屏亮度组成)
  - [2.2 环境光反射计算](#22-环境光反射计算)
  - [2.3 对比度比（Contrast Ratio）](#23-对比度比contrast-ratio)
  - [2.4 典型数值计算](#24-典型数值计算)
- [三、感知仿真公式推导](#三感知仿真公式推导)
  - [3.1 归一化模型](#31-归一化模型)
  - [3.2 公式展开](#32-公式展开)
  - [3.3 正确的感知模型](#33-正确的感知模型)
  - [3.4 代码实现](#34-代码实现)
- [四、完整的物理仿真流程](#四完整的物理仿真流程)
  - [4.1 输入输出关系](#41-输入输出关系)
  - [4.2 处理管道](#42-处理管道)
- [五、关键技术细节](#五关键技术细节)
  - [5.1 伽马校正处理](#51-伽马校正处理)
  - [5.2 高度对齐](#52-高度对齐)
  - [5.3 文字标注](#53-文字标注)
- [六、使用示例](#六使用示例)
  - [6.1 基本用法](#61-基本用法)
  - [6.2 参数说明](#62-参数说明)
  - [6.3 不同场景的典型参数](#63-不同场景的典型参数)
- [七、模型局限性](#七模型局限性)
  - [7.1 简化假设](#71-简化假设)
  - [7.2 改进方向](#72-改进方向)
- [八、总结](#八总结)

---

## 一、问题背景

当我们在不同环境光条件下观看显示屏时，人眼感知到的图像质量会发生显著变化：

- **暗室环境**（0 lux）：对比度最高，黑色纯净
- **办公室**（500 lux）：轻微泛白
- **窗边**（5000-25000 lux）：明显泛白，对比度下降
- **户外强光**（80000+ lux）：严重泛白，细节丢失

这个脚本的目标是**在屏幕上仿真这种物理现象**，帮助我们评估图像增强算法在不同光照条件下的实际效果。

---

## 二、核心物理模型

### 2.1 显示屏亮度组成

显示屏在环境光下的**总出射亮度**由两部分组成：

```
L_total(x) = L_emitted(x) + L_reflected
```

其中：
- `L_emitted(x)`：屏幕自发光亮度（位置 x 处）
- `L_reflected`：环境光反射亮度（均匀分布）

### 2.2 环境光反射计算

根据**朗伯反射定律**（Lambertian reflection）：

```
L_reflected = (E_ambient × ρ) / π
```

**推导过程**：

1. **环境照度** `E_ambient`（单位：lux = lumens/m²）
2. **表面反射率** `ρ`（reflectance，典型值 0.015 表示 1.5% 的 AR 涂层）
3. **朗伯体特性**：均匀漫反射，光强分布在半球面上

根据辐射度学：
```
Luminance (nit) = Illuminance (lux) × Reflectance / π
```

**为什么除以 π？**

朗伯反射面的辐射亮度公式：
```
L = (E × ρ) / π
```

这是因为朗伯体在半球面上的积分：
```
∫∫ cos(θ) dΩ = π steradians
```

### 2.3 对比度比（Contrast Ratio）

**传统定义**：
```
CR = L_white / L_black
```

在环境光下：
```
L_white = L_display_max + L_reflected
L_black = 0 + L_reflected = L_reflected

CR = (L_display_max + L_reflected) / L_reflected
   = L_display_max / L_reflected + 1
```

代码实现（第 53-63 行）：

```python
def compute_contrast_ratio(lux: float, display_nits: float, reflectance: float) -> float:
    """Compute on-screen contrast ratio (full-on/full-off) under given ambient illuminance.
    
    Traditional definition: CR = (L_display_max + L_reflected) / L_reflected
    """
    if lux <= 0:
        return float("inf")  # 暗室，无限对比度
    
    l_reflected = lux * reflectance / math.pi  # 反射亮度
    
    if l_reflected <= 0:
        return float("inf")
    
    return (display_nits + l_reflected) / l_reflected
```

### 2.4 典型数值计算

假设 `display_nits = 1000`，`reflectance = 0.015`：

| 环境光 (lux) | L_reflected (nit) | CR | 效果描述 |
|-------------|------------------|-----|---------|
| 0 | 0 | ∞ | 完美黑位 |
| 500 | 2.39 | 419:1 | 轻微泛白 |
| 5000 | 23.87 | 42.9:1 | 明显泛白 |
| 25000 | 119.37 | 9.38:1 | 严重泛白 |
| 80000 | 381.97 | 3.62:1 | 极度泛白 |

---

## 三、感知仿真公式推导

### 3.1 归一化模型

设屏幕输出图像为 `I(x) ∈ [0, 1]`（归一化亮度）

**实际物理亮度**：
```
L(x) = I(x) × L_display_max + L_reflected
```

**归一化到显示范围**：
```
L_max = L_display_max + L_reflected  (白色总亮度)
L_min = L_reflected                   (黑色总亮度)

I_perceived(x) = (L(x) - L_min) / (L_max - L_min)
```

### 3.2 公式展开

```
I_perceived(x) = [I(x) × L_display_max + L_reflected - L_reflected] / [L_display_max + L_reflected - L_reflected]

               = [I(x) × L_display_max] / L_display_max

               = I(x)
```

**等等！** 这样推导得出的是原始图像，没有体现对比度损失。

### 3.3 正确的感知模型

实际上，人眼会**自适应**到新的黑位和白位。正确的模型是：

**在 CR 降低时，感知对比度压缩**：

```
I_perceived(x) = I(x) × (CR - 1) / CR + 1 / CR
```

**推导**：

1. 原始动态范围：`[0, 1]`
2. 新动态范围：`[1/CR, 1]`（黑位抬高到 1/CR）
3. 线性映射：
   ```
   I_perceived = I × (1 - 1/CR) + 1/CR
               = I × (CR-1)/CR + 1/CR
   ```

**物理意义**：
- 当 `CR → ∞`：`I_perceived → I`（无损失）
- 当 `CR = 1`：`I_perceived → 1`（全白，无对比度）
- 当 `CR = 3.62`（80000 lux）：
  ```
  I_perceived = I × 0.724 + 0.276
  ```
  - 黑色 (I=0) → 0.276（严重泛白）
  - 白色 (I=1) → 1.0
  - 中间灰 (I=0.5) → 0.638

### 3.4 代码实现

```python
def apply_ambient_perception(img_float: np.ndarray, contrast_ratio: float) -> np.ndarray:
    """Apply the ambient-light wash-out model to a normalised [0, 1] image.
    
    Simplified model (no gamma encode/decode):
      perceived = image * (CR-1)/CR + 1/CR
    """
    if not math.isfinite(contrast_ratio):
        return np.clip(img_float, 0.0, 1.0)  # 暗室，无变化
    
    cr = contrast_ratio
    perceived = img_float * ((cr - 1.0) / cr) + 1.0 / cr
    
    return np.clip(perceived, 0.0, 1.0)
```

---

## 四、完整的物理仿真流程

### 4.1 输入输出关系

```
原始图像 I_original [0,1]
    ↓
环境光仿真
    ↓
感知图像 I_perceived = I × (CR-1)/CR + 1/CR
    ↓
量化到 uint8 [0, 255]
```

### 4.2 处理管道

#### 模式 1：Pair Mode（对比模式）

```python
# 第 166-207 行
for lux in lux_levels:
    cr = compute_contrast_ratio(lux, display_nits, reflectance)
    
    # 原始图像在环境光下
    orig_perc = apply_ambient_perception(orig_f, cr)
    
    # 增强图像在环境光下
    enh_perc = apply_ambient_perception(enh_f, cr)
    
    # 生成对比图：[原图(暗室) | 原图@lux | 增强图@lux]
    comp = build_comparison_image(raw_orig_u8, to_uint8(orig_perc), to_uint8(enh_perc), ...)
```

**输出布局**：

```
┌──────────────────────────────────────────────────┐
│ Original(no ambient) | Original@lux | Enhanced@lux │
├──────────────────────────────────────────────────┤
│     [图像]           |   [图像]     |   [图像]     │
└──────────────────────────────────────────────────┘
```

#### 模式 2：Composite Mode（综合模式）

```python
# 第 210-287 行
for lux in lux_levels:
    cr = compute_contrast_ratio(lux, display_nits, reflectance)
    
    # 左列：输入@lux
    left_cell = resize_cell(to_uint8(apply_ambient_perception(orig_f, cr)))
    
    # 中列：原图（暗室）
    raw_cell = resize_cell(raw_orig_u8)
    
    # 右列：增强@lux
    right_cell = resize_cell(to_uint8(apply_ambient_perception(enh_f, cr)))
    
    # 拼接一行
    row = hstack([left, sep, raw, sep, right])
    
# 垂直拼接所有 lux 行
composite = vstack([row_5000, row_25000, row_45000, row_80000])
```

**输出布局**：

```
┌─────────────────────────────────────────┐
│ Input@5000 | Original | Enhanced@5000   │
├─────────────────────────────────────────┤
│ Input@25000| Original | Enhanced@25000  │
├─────────────────────────────────────────┤
│ Input@45000| Original | Enhanced@45000  │
├─────────────────────────────────────────┤
│ Input@80000| Original | Enhanced@80000  │
└─────────────────────────────────────────┘
```

#### 模式 3：Simple/Matched Mode（批量模式）

```python
# 第 400-453 行
for lux in lux_levels:
    cr = compute_contrast_ratio(lux, display_nits, reflectance)
    
    # 保存 input@lux
    perceived_input = apply_ambient_perception(img_f, cr)
    imwrite(output_root / f"input_{lux}lux" / name, perceived_input)
    
    # 保存 enhanced@lux
    if enhanced_dir exists:
        perceived_enh = apply_ambient_perception(enh_f, cr)
        imwrite(output_root / f"enhanced_{lux}lux" / name, perceived_enh)
        
        # 保存对比图
        comp = build_comparison_image(...)
        imwrite(output_root / f"comparison_{lux}lux" / name, comp)
```

---

## 五、关键技术细节

### 5.1 伽马校正处理

**当前实现**：**不使用伽马校正**（简化模型）

```python
# 第 66-67 行
GAMMA = 2.2
INV_GAMMA = 1.0 / GAMMA

# 但实际没有使用，直接在 sRGB 空间操作
def apply_ambient_perception(img_float: np.ndarray, contrast_ratio: float):
    perceived = img_float * ((cr - 1.0) / cr) + 1.0 / cr
    return np.clip(perceived, 0.0, 1.0)
```

**为什么不用伽马？**

1. **简化计算**：线性光空间更准确，但需要额外转换
2. **sRGB 近似**：现代显示器已经做了伽马校正，sRGB 值近似线性
3. **相对比较**：我们关心的是原图 vs 增强图的**相对差异**，不是绝对物理精度

**完整模型（如果需要）**：

```python
# 1. sRGB → 线性光
def srgb_to_linear(img):
    return np.where(img <= 0.04045, img/12.92, ((img+0.055)/1.055)**2.4)

# 2. 在线性光空间应用环境光模型
L_linear = srgb_to_linear(img_float)
L_perceived = L_linear * (CR-1)/CR + 1/CR

# 3. 线性光 → sRGB
def linear_to_srgb(img):
    return np.where(img <= 0.0031308, img*12.92, 1.055*img**(1/2.4)-0.055)

return linear_to_srgb(L_perceived)
```

### 5.2 高度对齐

不同图像可能有不同分辨率，需要对齐：

```python
# 第 111-123 行
def build_comparison_image(...):
    h_o, w_o = raw_orig_u8.shape[:2]
    
    # 调整 perceived_orig 高度
    h_po, w_po = perceived_orig_u8.shape[:2]
    if h_po != h_o:
        new_w = int(round(w_po * (h_o / h_po)))
        perceived_orig_u8 = cv2.resize(perceived_orig_u8, (new_w, h_o))
    
    # 同样调整 perceived_enh
    ...
```

**保持宽高比**：
```
new_w = w_old × (h_target / h_old)
```

### 5.3 文字标注

```python
# 第 91-98 行
def _put_centered_text(canvas, text, x_left, x_right):
    font_scale = 0.8
    thickness = 2
    
    # 获取文字尺寸
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    
    # 计算居中位置
    cx = (x_left + x_right) // 2
    org = (cx - tw // 2, TOP_BAR_HEIGHT // 2 + th // 2)
    
    cv2.putText(canvas, text, org, FONT, font_scale, (0, 0, 0), thickness)
```

---

## 六、使用示例

### 6.1 基本用法

```bash
python scripts/simulate_ambient_viewing.py \
    --input ./test/input \
    --enhanced ./test/output/stage2/80000lux \
    --output ./test/simulate_viewing \
    --lux 5000 25000 45000 80000 \
    --display_nits 1000 \
    --reflectance 0.015
```

### 6.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 原始图像文件夹 |
| `--enhanced` | 可选 | 增强图像文件夹（pair 模式） |
| `--enhanced_dir` | 可选 | 包含 lux 子文件夹的目录（composite 模式） |
| `--output` | 必填 | 输出根目录 |
| `--lux` | 5000 25000 45000 80000 | 环境光照度（lux），可多个 |
| `--display_nits` | 1000 | 显示器峰值亮度（nit） |
| `--reflectance` | 0.015 | 屏幕反射率（AR 涂层约 1.5%） |
| `--composite` | False | 生成综合对比图（4行×3列） |

### 6.3 不同场景的典型参数

**车载显示屏**：
```bash
--display_nits 1000 --reflectance 0.015 --lux 5000 25000 80000
```

**手机（户外）**：
```bash
--display_nits 1500 --reflectance 0.02 --lux 10000 50000 100000
```

**室内显示器**：
```bash
--display_nits 400 --reflectance 0.03 --lux 300 500 1000
```

---

## 七、模型局限性

### 7.1 简化假设

1. **朗伯反射**：实际屏幕可能有镜面反射成分
2. **均匀环境光**：实际可能是方向性光源
3. **忽略眩光**：没有模拟光晕、眩光效果
4. **无伽马校正**：在 sRGB 空间直接操作
5. **单色模型**：没有考虑环境光色温对色彩的影响

### 7.2 改进方向

```python
# 更完整的模型可以加入：

# 1. 镜面反射（specular component）
L_total = L_diffuse + L_specular

# 2. 角度依赖
L_reflected(θ) = (E × ρ × cos(θ)) / π

# 3. 人眼自适应
I_adapted = tone_map(I_perceived, adaptation_luminance)

# 4. 色度影响（环境光色温）
RGB_perceived = apply_chromatic_adaptation(RGB, white_point)

# 5. 局部适应
L_local_adapted = apply_local_tone_mapping(L_perceived)
```

---

## 八、总结

这个脚本通过**简化的物理模型**实现了：

✅ **对比度损失仿真**：`CR = (L_max + L_refl) / L_refl`

✅ **黑位抬高**：`I_perceived = I × (CR-1)/CR + 1/CR`

✅ **多 lux 级别**：批量处理不同环境光条件

✅ **可视化对比**：原图 vs 增强图在相同环境光下

✅ **三种模式**：pair / composite / matched

### 核心价值

- 评估增强算法在**真实使用场景**下的效果
- 避免只在暗室中优化，导致户外效果差
- 为车载/手机等强光场景提供仿真工具
- 快速验证不同显示参数（nits、reflectance）的影响

这是一个**工程实用**的简化模型，在准确性和计算效率之间取得了良好的平衡！
