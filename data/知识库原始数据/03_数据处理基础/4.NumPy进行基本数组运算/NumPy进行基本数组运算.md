# NumPy 进行基本数组运算

① 底稿：《Python NumPy 入门教程：从数组创建到广播运算，快速掌握高效数值计算》  
　　网址：https://blog.csdn.net/yuanpan/article/details/160544416

② 参考文章：《Python NumPy数学函数深度剖析：高性能科学计算的基石与进阶》  
　　网址：https://blog.csdn.net/2401_83754820/article/details/160058368

③ 参考文章：《Python NumPy 入门教程（3）花式索引和索引技巧》  
　　网址：https://blog.csdn.net/weixin_42107409/article/details/151954179

④ 参考文章：《NumPy基础入门（3）花式索引和索引技巧》  
　　网址：https://blog.csdn.net/weixin_42105744/article/details/103931289

⑤ 参考文章：《「Python数据科学」一文搞懂NumPy中的广播机制》  
　　网址：https://blog.csdn.net/dqrcsc/article/details/144586555

> **适合人群**：已完成"格式转换（JSON/CSV 读写）"章节，掌握 Python 基础和 Pandas 基本操作的初学者。本节时长约 1 小时，目标是掌握 NumPy 数组的创建、索引、批量运算、广播和聚合统计，能够用 NumPy 高效处理数值数据。（①）

## 一、为什么需要 NumPy？（①）

上节我们用 Pandas 处理表格型数据，但 Pandas 底层依赖的就是 NumPy。当你需要处理大规模数值计算（如矩阵运算、向量操作、数据归一化）时，NumPy 比 Python 原生列表快得多。（①）

**核心优势**：（①+②）

| 对比项 | Python 原生 list | NumPy ndarray |
|--------|-----------------|---------------|
| 运算方式 | 逐个元素循环 | 整批矢量化运算 |
| 速度 | 慢（Python 解释器开销） | 快（底层 C 实现） |
| 内存 | 存储对象引用，开销大 | 连续内存块，紧凑高效 |
| 维度 | 一维 | 支持任意维度 |

> **一句话总结**：NumPy 让大量数值操作更快、更整洁、更适合批量处理。（①）

## 二、数组创建与基本属性（①）

### 2.1 导入与创建数组（①）

```python
import numpy as np

# 从列表创建
arr = np.array([10, 20, 30, 40])

# 创建全 0 / 全 1 数组
zeros = np.zeros((2, 3))      # 2行3列
ones = np.ones((3, 2))        # 3行2列

# 创建连续数字（类似 range）
arr = np.arange(1, 10)        # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 创建等间距数组
arr = np.linspace(0, 1, 5)      # [0.  , 0.25, 0.5 , 0.75, 1.  ]
```

### 2.2 查看数组属性（①）

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)    # (2, 3)  → 2行3列
print(arr.dtype)    # int64   → 元素类型
print(arr.ndim)     # 2       → 维度数
```

> **shape 是最常用的属性**，后续所有操作都要先确认数组形状。（①）

### 2.3 改变数组形状：reshape（①）

```python
arr = np.arange(1, 13)          # 1维，12个元素
matrix = arr.reshape(3, 4)       # 变为 3行4列
print(matrix)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
```

> `reshape` 不会改变原始数组的数据，只是改变"看待数据的方式"。元素总数必须一致（3×4=12）。（①）

## 三、索引与切片（①+③+④）

### 3.1 基础索引（①）

```python
arr = np.arange(1, 13).reshape(3, 4)

print(arr[1, 2])        # 第2行第3列 → 7
print(arr[:, 1])        # 第2列全部 → [2, 6, 10]
print(arr[0:2, 1:3])    # 前2行，第2-3列
# [[2 3]
#  [6 7]]
```

> 索引规则：**先行后列**，`arr[行, 列]`。切片时左闭右开。（①）

### 3.2 布尔索引：按条件筛选（③+④）

```python
scores = np.array([88, 92, 79, 95, 90])

# 筛选出大于等于90分的
high_scores = scores[scores >= 90]
print(high_scores)      # [92 95 90]

# 得到布尔数组
print(scores >= 90)     # [False  True False  True  True]
```

> 布尔索引在数据清洗中非常实用，比如"选出所有大于某个阈值的样本"。（③）

### 3.3 整数数组索引：花式索引（③+④）

```python
arr = np.array([10, 20, 30, 40, 50])

# 用索引数组提取特定位置
indices = np.array([0, 2, 4])
print(arr[indices])     # [10 30 50]

# 二维数组的花式索引
matrix = np.arange(12).reshape(3, 4)
print(matrix[[0, 2], [1, 3]])   # 取 (0,1)=1 和 (2,3)=11 → [1, 11]
```

> 花式索引返回的是**副本**（新数组），修改不会影响原数组。（③）

## 四、批量运算与广播机制（①+⑤）

### 4.1 整批运算：无需循环（①）

```python
prices = np.array([15, 20, 35, 40])

# 所有元素统一操作
print(prices * 0.9)     # [13.5 18.  31.5 36. ]
print(prices + 10)      # [25 30 45 50]
print(prices / 2)       # [ 7.5 10.  17.5 20. ]
```

> 这种"整批运算"的能力是 NumPy 最核心的优势，底层由 C 实现，速度远超 Python 循环。（①+②）

### 4.2 广播机制：不同形状也能运算（①+⑤）

广播是 NumPy 的自动"形状对齐"机制。当两个数组形状不同时，NumPy 会尝试将较小的数组"拉伸"以匹配较大数组的形状。（⑤）

**最简单的广播：数组与标量**

```python
a = np.array([1, 2, 3])
result = a * 2
# 等价于 a * [2, 2, 2]，标量 2 被广播到数组的每个位置
```

**二维广播：向量与矩阵**

```python
scores = np.array([
    [80, 85, 90],
    [75, 88, 92]
])  # shape (2, 3)

bonus = np.array([5, 0, 3])  # shape (3,)

result = scores + bonus
print(result)
# [[85 85 93]
#  [80 88 95]]
```

> `bonus` 的形状是 `(3,)`，NumPy 会将其视为 `(1, 3)`，然后沿着行方向（第 0 维）拉伸为 `(2, 3)`，与 `scores` 对齐后逐元素相加。（⑤）

**广播规则速查**：（⑤）

1. 维度不同时，在较小数组的**左侧**补 1，直到维度数相同
2. 逐维度比较，要么相等，要么其中一个为 1
3. 不满足则报错：`ValueError: operands could not be broadcast together`

```python
# 错误示例：形状不兼容
arr_a = np.array([[1, 2], [3, 4]])      # (2, 2)
arr_b = np.array([10, 20, 30])           # (3,)
# arr_a + arr_b  → 报错！最右维度 2 ≠ 3，且都不为1
```

## 五、聚合统计：sum、mean、max、min（①+②）

### 5.1 全局统计（①）

```python
sales = np.array([
    [1200, 1500, 1800],
    [1000, 1600, 1700],
    [900, 1400, 2100]
])

print(sales.sum())      # 总销售额
print(sales.mean())     # 平均值
print(sales.max())      # 最大值
print(sales.min())      # 最小值
```

### 5.2 按轴统计：理解 axis（①+②）

`axis` 是 NumPy 中最容易混淆的概念。记住这个口诀：（①）

- `axis=0`：沿着**行**方向压缩 → 得到**每列**的结果
- `axis=1`：沿着**列**方向压缩 → 得到**每行**的结果

```python
print(sales.sum(axis=0))    # 按列求和 → [3100, 4500, 5600]
print(sales.mean(axis=1))   # 按行求均值 → [1500., 1433.33, 1466.67]
```

### 5.3 保持维度：keepdims（②）

默认情况下，聚合操作会**移除**被压缩的维度。使用 `keepdims=True` 可以保留该维度（大小变为 1），方便后续广播。

```python
row_mean = sales.mean(axis=1, keepdims=True)
print(row_mean.shape)       # (3, 1)

# 利用 keepdims 进行广播：每个元素减去所在行的均值
centered = sales - row_mean
print(np.round(centered, 2))
```

## 六、条件筛选与数据变换（①+③）

### 6.1 条件替换：np.where（①）

```python
scores = np.array([88, 92, 79, 95, 90])

# 大于等于90为"A"，否则为"B"
levels = np.where(scores >= 90, "A", "B")
print(levels)     # ['B' 'A' 'B' 'A' 'A']
```

### 6.2 数据裁剪：np.clip（②）

```python
data = np.array([-5, 10, 25, 100, 3])

# 限制在 [0, 20] 范围内
clipped = np.clip(data, 0, 20)
print(clipped)    # [ 0 10 20 20  3]
```

### 6.3 处理缺失值：np.nan_to_num（②）

```python
data = np.array([1, 2, np.nan, 4, np.inf])
cleaned = np.nan_to_num(data, nan=0.0)
print(cleaned)    # [ 1.  2.  0.  4. inf]
```

> 实际数据处理中，建议先用 Pandas 清洗缺失值，再用 NumPy 做数值运算。（②）

## 七、随机数生成（①）

```python
# 推荐新写法：使用 Generator（可复现）
rng = np.random.default_rng(seed=42)

# 生成 2行3列 的随机整数 [0, 10)
random_ints = rng.integers(0, 10, size=(2, 3))

# 生成标准正态分布的随机数
random_floats = rng.normal(loc=0, scale=1, size=(3, 3))
```

> `seed=42` 让结果可复现，在实验和调试时非常重要。（①）

## 八、矩阵运算（①）

### 8.1 逐元素乘法 vs 矩阵乘法（①）

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 逐元素相乘
print(a * b)
# [[ 5 12]
#  [21 32]]

# 矩阵乘法（线性代数中的点积）
print(a @ b)
# [[19 22]
#  [43 50]]
```

> **新手最常踩的坑**：把 `*` 当成矩阵乘法。记住：`*` 是逐元素，`@` 才是矩阵乘法。（①）

## 九、综合实战演练（①）

请在你的 Python 环境中完成以下操作：

### 场景：学生成绩统计与分析

```python
import numpy as np

# 1. 创建成绩数据：3门课 × 5个学生
scores = np.array([
    [85, 90, 78, 92, 88],   # 数学
    [76, 88, 95, 80, 85],   # 英语
    [90, 85, 82, 78, 91]    # 物理
])  # shape (3, 5)

print("原始成绩：")
print(scores)

# 2. 每门课加 5 分平时成绩（广播）
final_scores = scores + 5
print("\n加平时成绩后：")
print(final_scores)

# 3. 统计每门课的平均分（axis=1，沿学生方向压缩）
course_mean = final_scores.mean(axis=1)
print("\n每门课平均分：", np.round(course_mean, 2))

# 4. 统计每个学生的总分（axis=0，沿课程方向压缩）
student_total = final_scores.sum(axis=0)
print("每个学生总分：", student_total)

# 5. 找出所有大于 90 分的成绩
high_scores = final_scores[final_scores > 90]
print("\n大于90分的成绩：", high_scores)

# 6. 将成绩转换为等级
grades = np.where(final_scores >= 90, "A",
          np.where(final_scores >= 80, "B", "C"))
print("\n成绩等级：")
print(grades)
```

## 十、常见问题与解决（①+②+⑤）

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| `reshape` 报错 | 元素总数不匹配 | 确认 `新形状各维度乘积 = 原数组大小` |
| 广播报错 `operands could not be broadcast` | 形状不兼容 | 从最右维度开始对比，确保相等或其中一个为 1 |
| `axis=0` 和 `axis=1` 结果搞反 | 对 axis 理解不清 | 记住：`axis=0` 压缩行得列结果，`axis=1` 压缩列得行结果 |
| 修改切片后原数组也变了 | 切片返回的是视图（共享内存） | 需要独立副本时用 `.copy()` |
| `*` 和 `@` 结果不同 | 混淆逐元素乘法和矩阵乘法 | `*` 逐元素，`@` 矩阵乘法 |
| 随机数每次运行结果不同 | 没有设置 seed | `rng = np.random.default_rng(seed=42)` |

## 十一、总结与后续学习路线（①+②）

恭喜你完成了 NumPy 基本数组运算章节的学习！你现在应该掌握了：

- ✅ NumPy 的核心优势：矢量化运算、连续内存、底层 C 实现
- ✅ 数组创建：`np.array`、`np.zeros`、`np.ones`、`np.arange`、`np.linspace`
- ✅ 数组属性：`shape`、`dtype`、`ndim`
- ✅ 索引切片：基础索引 `[行, 列]`、布尔索引、花式索引
- ✅ 批量运算：无需循环，直接对整个数组操作
- ✅ 广播机制：不同形状数组的自动对齐规则
- ✅ 聚合统计：`sum`、`mean`、`max`、`min` + `axis` + `keepdims`
- ✅ 条件操作：`np.where`、`np.clip`
- ✅ 随机数生成：可复现的 `default_rng`
- ✅ 矩阵运算：`*` 逐元素 vs `@` 矩阵乘法

> **下一步预告**：在下一节 **"正则表达式进行文本预处理"** 中，我们将学习如何用 Python 内置的 `re` 模块从非结构化文本中提取结构化信息。随后进入 **PyTorch 基础** 板块时，你会发现 PyTorch 的 **Tensor（张量）** 与 NumPy 的 **ndarray** 概念高度相似——今天学的索引、切片、广播、聚合等操作，在 PyTorch 中几乎一模一样。现在，试着用 NumPy 处理一份自己的数值数据吧！