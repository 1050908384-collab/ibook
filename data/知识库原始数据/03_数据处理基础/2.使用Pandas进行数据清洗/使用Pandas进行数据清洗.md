# 使用 Pandas 进行数据清洗

① 底稿：《Pandas数据清洗实战：从脏数据到可用数据》  
　　网址：https://blog.csdn.net/weixin_54482667/article/details/159381750

② 参考文章：《数据清洗：缺失值、重复值、异常值处理》  
　　网址：https://blog.csdn.net/m0_51069462/article/details/160083221

③ 参考文章：《数据清洗第一课：缺失值、重复值和异常值怎么处理》  
　　网址：https://blog.csdn.net/weixin_61731209/article/details/162586457

> **适合人群**：已完成"环境准备"章节，在 Python 环境中成功安装并导入 Pandas 的初学者。本节时长约 1 小时，目标是掌握使用 Pandas 处理缺失值、重复值、异常值、数据类型转换和字符串规范化的核心方法，能将一份"脏数据"清洗为可用于分析或训练的干净数据。（①+③）

## 一、为什么需要数据清洗？（①+③）

真实世界中的数据很少是完美的。采集过程中可能出现设备故障、用户漏填、系统传输错误、人工录入失误等问题，导致数据中存在缺失、重复、异常、格式混乱等质量问题。（①）

如果直接把脏数据喂给模型，会产生"垃圾进，垃圾出"（Garbage In, Garbage Out）的结果。数据清洗的目标是让数据变得**完整、一致、准确、可用**。（①+③）

常见的数据质量问题包括：（③）

| 问题类型 | 表现 | 对后续分析的影响 |
|---------|------|----------------|
| **缺失值** | 某些字段没有记录 | 无法计算，或填充不当会引入偏差 |
| **重复值** | 同一条样本出现多次 | 统计结果翻倍，模型对重复样本过度学习 |
| **异常值** | 数值明显偏离合理范围 | 拉偏均值、扭曲拟合方向 |
| **格式混乱** | 日期、金额、类别写法不统一 | 同义值被当成不同类别 |
| **标签错误** | 目标值本身标错了 | 模型认真学习错误答案 |

本节聚焦前四类问题的处理，标签错误排查将在后续课程中涉及。（③）

## 二、数据加载与初步探索（①）

任何清洗工作的第一步，都是先加载数据、了解数据结构，再制定清洗策略。（①）

```python
import pandas as pd
import numpy as np

# 加载数据（后续"格式转换"章节会详细讲解 read_csv 的参数）
df = pd.read_csv('data.csv', encoding='utf-8')

# 查看数据的基本信息
print(df.shape)        # (行数, 列数)
print(df.columns)      # 列名列表
print(df.dtypes)       # 每列的数据类型
print(df.head())       # 前5行预览
print(df.tail())       # 后5行预览
```

> **为什么先看 `dtypes`？** 因为从 CSV 读取的数据，数字可能被识别为字符串（`object`），日期可能被识别为普通文本。了解当前类型，才能判断是否需要转换。（①）

```python
# 快速统计数值列的分布
print(df.describe())

# 查看每列的缺失值数量
print(df.isnull().sum())
```

## 三、缺失值处理（①+②+③）

缺失值是数据清洗中最常见的问题。在 Pandas 中，缺失值通常用 `NaN`（Not a Number）表示。（②）

### 3.1 检测缺失值（①+②）

```python
# 查看每列缺失值数量和缺失率
missing_info = pd.DataFrame({
    '缺失数量': df.isnull().sum(),
    '缺失率': (df.isnull().sum() / len(df)).round(4)
})
print(missing_info[missing_info['缺失数量'] > 0])

# 查看哪些行有缺失值
print(df[df.isnull().any(axis=1)])
```

> **缺失本身可能是信号**：用户没填收入，可能代表对隐私敏感；某项医疗检测没做，可能因为患者看起来健康。"没记录"本身也是一条信息，不要盲目删除。（③）

### 3.2 处理策略：删、填、标记（①+②+③）

**策略一：删除缺失值**

```python
# 删除包含任何缺失值的行
df_clean = df.dropna()

# 只删除特定列缺失的行
df_clean = df.dropna(subset=['age', 'salary'])

# 删除缺失值超过阈值的行（至少保留2个非缺失值）
df_clean = df.dropna(thresh=2)
```

> **什么时候删除？** 缺失比例极低、且该列对分析不重要时。如果缺失率高或缺失本身有业务含义，删除会丢失信息。（③）

**策略二：填充缺失值**

```python
# 用常数填充
df['discount'] = df['discount'].fillna(0)          # 金额缺失填0（表示无折扣）
df['city'] = df['city'].fillna('未知')              # 类别缺失填"未知"

# 用统计量填充
df['age'] = df['age'].fillna(df['age'].median())   # 数值型推荐中位数（抗异常值）
df['category'] = df['category'].fillna(df['category'].mode()[0])  # 分类型用众数

# 时间序列常用：向前/向后填充
df['value'] = df['value'].ffill()   # 用上一个有效值填充
df['value'] = df['value'].bfill()   # 用下一个有效值填充

# 线性插值（数据有规律变化时）
df['value'] = df['value'].interpolate(method='linear')
```

**策略三：标记缺失**

```python
# 如果缺失本身可能是信号，新增一列标记
df['income_missing'] = df['income'].isnull().astype(int)
df['income'] = df['income'].fillna(df['income'].median())
```

> **不同数据类型的填充策略速查**：（②）

| 数据类型 | 推荐填充方法 | 示例 |
|---------|-------------|------|
| 数值（近似正态分布） | 均值 | `fillna(mean)` |
| 数值（偏态分布） | 中位数 | `fillna(median)` |
| 分类 / 字符串 | 众数或"未知" | `fillna(mode[0])` |
| 时间序列 | 前向填充 / 线性插值 | `method='ffill'` |
| 业务金额类 | 0 或特定业务值 | `fillna(0)` |

> **⚠️ 重要警告：避免数据泄漏**  
> 不要在拆分训练集/测试集之前，用全量数据计算填充值。正确做法是先拆分，**只在训练集上学习填充规则**（如计算中位数），再应用到测试集。否则测试集的信息会泄漏到训练阶段，导致评估结果过于乐观。（③）

## 四、重复值处理（①+②+③）

重复数据会导致统计结果偏差，特别是在计数和求和时影响明显。（①）

### 4.1 检测重复值（①+②）

```python
# 检测完全重复的行
print(df.duplicated().sum())

# 基于指定列检测重复
print(df.duplicated(subset=['user_id']).sum())

# 查看重复行的具体内容
print(df[df.duplicated(keep=False)])
```

### 4.2 处理重复值（①+②）

```python
# 删除完全重复的行（保留第一次出现）
df_clean = df.drop_duplicates()

# 保留最后一次出现
df_clean = df.drop_duplicates(keep='last')

# 基于指定列去重（如一个订单不应出现多次）
df_clean = df.drop_duplicates(subset=['order_id'], keep='first')

# 去重后重置索引
df_clean = df_clean.reset_index(drop=True)
```

> **结合业务主键判断**：同一个 `user_id` 出现多次，不一定是数据错误——可能是同一用户在不同时间下了多笔订单。去重前应确认业务主键（如 `user_id + order_id + order_time`），避免误删真实数据。（③）

## 五、异常值检测与处理（①+②+③）

异常值是指明显偏离正常范围的数据点。它们可能是录入错误，也可能是真实的极端情况（如大客户的高消费）。（③）

### 5.1 统计方法定位异常值（①+②）

**方法一：IQR 四分位距法（最常用，不依赖分布）**

```python
Q1 = df['amount'].quantile(0.25)   # 第一四分位数
Q3 = df['amount'].quantile(0.75)   # 第三四分位数
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
outliers = (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
print(f"异常值数量: {outliers.sum()}")
```

**方法二：Z-Score 法（适合近似正态分布的数据）**

```python
# 先确保 amount 为数值型；缺失值会保留为 NaN，不会被标记为异常值
mean = df['amount'].mean()
std = df['amount'].std(ddof=0)
z_scores = np.abs((df['amount'] - mean) / std)
outliers = z_scores > 3
```

> **其他方法**：3σ 原则、箱线图可视化、DBSCAN 聚类、孤立森林（Isolation Forest）等，适用于更复杂的场景。初学者掌握 IQR 法即可应对大多数情况。（②）

### 5.2 处理策略：统计定位，业务决策（③）

| 情况 | 判断 | 处理方式 |
|------|------|---------|
| 年龄 = 300 | 不可能，录入错误 | 标为缺失或删除 |
| 收入 = -5000 | 不可能，数据错误 | 标为缺失或删除 |
| 单笔消费 = 50 万 | 可能，大客户 | 结合业务上下文决定 |

```python
# 策略一：删除明显的错误数据
df_clean = df[(df['age'] >= 0) & (df['age'] <= 120)]

# 策略二：截断到合理范围（保留样本但限制极端值影响）
df['amount'] = df['amount'].clip(lower=0, upper=df['amount'].quantile(0.99))

# 策略三：用中位数替换异常值
median_amount = df.loc[~outliers, 'amount'].median()
df.loc[outliers, 'amount'] = median_amount
```

> **不要一刀切**：信用卡欺诈检测中，异常值恰恰是目标；电商大促的销售额异常是正常业务。先用统计方法定位，再结合业务知识决定处理方式。（③）

## 六、数据类型转换与字符串规范化（①）

### 6.1 数据类型转换（①）

```python
# 将字符串日期转为 datetime 类型
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# 从日期中提取特征
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['weekday'] = df['order_date'].dt.day_name()

# 确保数值列为正确类型
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['quantity'] = df['quantity'].astype('Int64')  # 可空整数类型

# 分类字段转为 category 类型（节省内存）
df['category'] = df['category'].astype('category')
```

> `errors='coerce'` 的作用：如果某行数据无法转换（如日期字段里混入了 `"invalid"`），Pandas 会将其设为 `NaN` 而不是报错中断程序。（①）

### 6.2 字符串规范化（①）

```python
# 去除前后空格
df['product_name'] = df['product_name'].str.strip()

# 统一大小写
df['category'] = df['category'].str.lower()

# 统一表示方式（映射字典）
payment_map = {
    'credit card': '信用卡',
    'credit_card': '信用卡',
    'alipay': '支付宝',
    'zfb': '支付宝',
    'wechat': '微信支付'
}
df['payment_method'] = df['payment_method'].map(
    lambda x: payment_map.get(x, x)
)
```

## 七、综合实战演练（①+②+③）

请在你的 Python 环境中完成以下操作，巩固本节所学。

### 场景：清洗一份模拟的用户数据

```python
import pandas as pd
import numpy as np

# 创建一份包含常见问题的模拟数据
df = pd.DataFrame({
    "user_id": [101, 102, 103, 104, 105, 101, 106],
    "age": [25, np.nan, 38, 300, 45, 25, 52],           # 缺失 + 异常
    "income": [8000, 12000, np.nan, 9500, -5000, 8000, 60000],  # 缺失 + 异常
    "city": ["北京", "上海", "广州", "深圳", "杭州", "北京", np.nan],
    "signup_date": ["2024-01-01", "2024-01-02", "2024-01-03",
                    "invalid", "2024-01-05", "2024-01-01", "2024-01-07"],
    "is_vip": ["yes", "NO", "Yes", "no", "YES", "yes", "No"]
})

print("原始数据：")
print(df)
print(f"\n原始形状: {df.shape}")
```

**步骤 1：处理重复值**

```python
# 基于 user_id 去重，保留第一次出现
df = df.drop_duplicates(subset=['user_id'], keep='first')
print(f"去重后形状: {df.shape}")
```

**步骤 2：处理缺失值**

```python
# age 用中位数填充
df['age'] = df['age'].fillna(df['age'].median())

# city 用"未知"填充
df['city'] = df['city'].fillna('未知')

# income 先标记缺失，再用中位数填充
df['income_missing'] = df['income'].isnull().astype(int)
df['income'] = df['income'].fillna(df['income'].median())
```

**步骤 3：处理异常值**

```python
# 年龄必须在合理范围
df = df[(df['age'] >= 0) & (df['age'] <= 120)]

# 收入截断到非负，并封顶到 99 分位数
df['income'] = df['income'].clip(lower=0, upper=df['income'].quantile(0.99))
```

**步骤 4：数据类型转换**

```python
# 日期转换，无效值变为 NaT
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')

# 统一 VIP 标识
vip_map = {'yes': 1, 'YES': 1, 'Yes': 1, 'no': 0, 'NO': 0, 'No': 0}
df['is_vip'] = df['is_vip'].map(vip_map)
```

**步骤 5：最终验证**

```python
print("\n清洗后数据：")
print(df)
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n数据类型:\n{df.dtypes}")
```

## 八、常见问题与解决（①+②+③）

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| `fillna()` 后仍有缺失 | 数据类型不匹配 | 确认列类型，字符串缺失和数值缺失处理方式不同 |
| `drop_duplicates()` 后数据变少很多 | 业务主键选择不当 | 检查是否误把合理重复当错误（如多笔订单） |
| 异常值处理后模型效果变差 | 删除了有价值的极端样本 | 先分析异常原因，再决定删除/截断/保留 |
| `pd.to_datetime()` 报错 | 日期格式不统一 | 使用 `errors='coerce'` 将无效值转为 NaT |
| 清洗后数据分布明显改变 | 过度清洗 | 保留原始副本，对比清洗前后的统计摘要 |

> **三大常见坑，务必避开**：（③）  
> 1. **先填充再拆分数据集** → 造成数据泄漏，评估结果虚高。应先拆分，只在训练集上学习填充规则。  
> 2. **过度清洗** → 把不顺眼的样本全删了，训练集过于理想化，模型上线后水土不服。  
> 3. **只洗特征，不查标签** → 标签本身错了，模型会认真学习错误答案。

## 九、总结与后续学习路线（①+②+③）

恭喜你完成了 Pandas 数据清洗章节的学习！你现在应该掌握了：

- ✅ 数据清洗的核心目标：让数据完整、一致、准确、可用
- ✅ 缺失值处理：检测（`isnull()`）、删除（`dropna()`）、填充（`fillna()`）、标记（新增列）
- ✅ 重复值处理：检测（`duplicated()`）、去重（`drop_duplicates()`），结合业务主键判断
- ✅ 异常值处理：IQR 法定位，业务知识决策，删除 / 截断 / 保留三种策略
- ✅ 数据类型转换：`pd.to_datetime()`、`pd.to_numeric()`、`astype('category')`
- ✅ 字符串规范化：`str.strip()`、`str.lower()`、映射字典统一表示
- ✅ 数据清洗中的关键安全意识：避免数据泄漏、避免过度清洗、检查标签质量

> **下一步预告**：在下一节 **"格式转换（JSON/CSV 读写）"** 中，我们将深入学习 `pd.read_csv()` 和 `pd.to_json()` 的参数配置，把清洗后的数据保存为标准化格式，以及从各种来源加载原始数据。随后还将进入 **"NumPy 进行基本数组运算"**，为后续 PyTorch 张量操作打下基础。现在，试着找一份自己的数据，用今天学到的方法清洗一遍吧！
