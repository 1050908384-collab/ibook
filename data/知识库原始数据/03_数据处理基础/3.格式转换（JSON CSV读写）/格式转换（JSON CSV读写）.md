# 格式转换（JSON/CSV 读写）

① 底稿：《Python中的json、csv、正则和包的管理》  
　　网址：https://blog.csdn.net/qq_44945384/article/details/160244761

② 参考文章：《Python 数据格式转换：从 JSON、XML 到 CSV、Excel 的无缝转换》  
　　网址：https://blog.csdn.net/shanwei_spider/article/details/150574128

③ 参考文章：《Holistic Tracking数据导出格式转换：CSV/JSON互转实战教程》  
　　网址：https://blog.csdn.net/weixin_42601547/article/details/156921813

④ 参考文章：《第37课：Python｜JSON/CSV/XML数据解析与序列化全套实战教程》  
　　网址：https://blog.csdn.net/searchb/article/details/161042445

> **适合人群**：已完成"使用 Pandas 进行数据清洗"章节，掌握 DataFrame 基本操作的初学者。本节时长约 0.5 小时，目标是掌握 JSON 和 CSV 两种数据格式的读写方法，能够将清洗后的数据保存到文件，也能从文件中加载数据供后续分析使用。（①+④）

## 一、为什么需要 JSON 和 CSV？（①+④）

在数据处理流程中，清洗后的数据需要持久化存储，或与其他系统交换。JSON 和 CSV 是最常用的两种格式：（①）

| 格式 | 特点 | 典型场景 |
|------|------|---------|
| **JSON** | 键值对结构，层级嵌套，人类可读 | API 接口数据、配置文件、模型训练参数 |
| **CSV** | 纯文本表格，一行一条记录，逗号分隔 | 数据集存储、Excel 兼容、数据库导入导出 |

上节我们用 Pandas 清洗了数据，本节学习如何把清洗结果保存为这两种格式，以及如何从它们加载数据。（①）

## 二、JSON 数据处理（①+④）

JSON（JavaScript Object Notation）是轻量级的数据交换格式。Python 内置 `json` 模块，无需安装。（①）

### 2.1 JSON 与 Python 类型的对应关系（①）

| JSON 类型 | Python 类型 |
|-----------|-------------|
| 对象 `{}` | 字典 `dict` |
| 数组 `[]` | 列表 `list` |
| 字符串 `"..."` | 字符串 `str` |
| 数字 | `int` / `float` |
| `true` / `false` | `True` / `False` |
| `null` | `None` |

> **注意**：JSON 的键名**必须用双引号**，字符串也必须是双引号。这是与 Python 字典最直观的区别。（④）

### 2.2 核心函数：带 s 与不带 s（①+④）

| 函数 | 作用 | 数据来源 |
|------|------|---------|
| `json.dumps(obj)` | Python 对象 → JSON **字符串** | 内存变量 |
| `json.loads(str)` | JSON **字符串** → Python 对象 | 内存变量 |
| `json.dump(obj, f)` | Python 对象 → 写入**文件** | 文件对象 |
| `json.load(f)` | 从**文件**读取 → Python 对象 | 文件对象 |

> **记忆技巧**：带 `s` 的是 **S**tring（字符串），不带 `s` 的是 **F**ile（文件）。（①）

### 2.3 常用参数（④）

```python
import json

data = {
    "name": "张三",
    "age": 25,
    "hobbies": ["阅读", "编程"],
    "is_student": False
}

# 序列化为格式化的 JSON 字符串
json_str = json.dumps(data, indent=4, ensure_ascii=False)
print(json_str)
```

关键参数说明：（④）

- `indent=4`：缩进 4 个空格，让输出可读
- `ensure_ascii=False`：允许输出中文（否则中文会被转义为 `\uXXXX`）
- `sort_keys=True`：按键名排序输出

### 2.4 读写 JSON 文件（①+④）

```python
# 写入 JSON 文件
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# 从 JSON 文件读取
with open("data.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

print(loaded_data["name"])  # 输出：张三
```

> **⚠️ 务必指定 `encoding='utf-8'`**，否则中文可能乱码。（④）

## 三、CSV 数据处理（①+④）

CSV（Comma-Separated Values）是纯文本表格格式。Python 内置 `csv` 模块。（①）

### 3.1 基于列表的读写（①）

```python
import csv

# 写入 CSV（列表形式）
data = [
    ["姓名", "年龄", "城市"],
    ["张三", 25, "北京"],
    ["李四", 30, "上海"]
]

with open("people.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)
```

> **⚠️ `newline=''` 必须加**：在 Windows 上写入 CSV 时，如果不加会产生多余的空行。（④）

```python
# 读取 CSV（列表形式）
with open("people.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
# 输出：['姓名', '年龄', '城市']
#       ['张三', '25', '北京']
#       ...
```

> 注意：`csv.reader` 读取的所有字段都是**字符串类型**，数值需要手动转换。（④）

### 3.2 基于字典的读写（推荐）（①+④）

字典形式通过表头映射数据，代码更健壮：

```python
# 写入 CSV（字典形式）
people = [
    {"name": "赵六", "age": 35, "city": "深圳"},
    {"name": "周七", "age": 32, "city": "成都"}
]

with open("people_dict.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()      # 必须先写表头！
    writer.writerows(people)
```

> **⚠️ `writeheader()` 不要忘记**，否则 CSV 没有列名行。（④）

```python
# 读取 CSV（字典形式）
with open("people_dict.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['name']} - {row['age']}岁 - {row['city']}")
```

### 3.3 处理特殊字符（④）

CSV 写入时会自动处理包含逗号、换行符或引号的字段，用双引号包裹：

```python
data = [
    ["产品", "描述"],
    ["笔记本", '14英寸, 轻薄本'],    # 包含逗号
    ["手机", '5G\n全网通']             # 包含换行
]

with open("special.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)
```

读取时会正确还原，无需额外处理。（④）

## 四、用 Pandas 读写 JSON/CSV（②+④）

上节学过的 Pandas 提供了更简洁的读写方式，适合处理 DataFrame：（②）

### 4.1 Pandas 读写 CSV

```python
import pandas as pd

# 读取 CSV 为 DataFrame
df = pd.read_csv("people_dict.csv", encoding="utf-8")

# 查看数据
print(df.head())

# 保存 DataFrame 为 CSV
df.to_csv("output.csv", index=False, encoding="utf-8")
```

> `index=False` 表示不保存行索引，否则 CSV 会多出一列无意义的数字索引。（②）

### 4.2 Pandas 读写 JSON

```python
# 读取 JSON 为 DataFrame（文件顶层必须是对象列表，例如后文生成的 people.json）
df = pd.read_json("people.json", encoding="utf-8")

# 保存 DataFrame 为 JSON
df.to_json("output.json", orient="records", force_ascii=False, indent=4)
```

> `orient="records"` 表示每行一个 JSON 对象，组成数组。这是最常见的 JSON 数据格式。（②）

## 五、JSON 与 CSV 互转（②+③+④）

在实际工作中，经常需要在两种格式间转换。

### 5.1 CSV 转 JSON（②+④）

```python
import csv
import json

def csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 将数值字符串转为合适类型
            if "age" in row:
                row["age"] = int(row["age"])
            data.append(row)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# 调用
csv_to_json("people_dict.csv", "people.json")
```

### 5.2 JSON 转 CSV（②+③）

```python
import json
import csv

def json_to_csv(json_file, csv_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 获取所有字段名（假设是列表中的字典）
    fieldnames = data[0].keys()
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# 调用
json_to_csv("people.json", "people_from_json.csv")
```

> **注意**：JSON 转 CSV 时，如果 JSON 是嵌套结构（如字典里包含列表），需要先扁平化才能写入 CSV。CSV 是二维表格，无法直接表示嵌套数据。（③）

## 六、综合实战演练（①+②+④）

请在你的 Python 环境中完成以下操作：

### 场景 1：创建并保存 JSON 配置文件

```python
import json

config = {
    "model_name": "bert-base-chinese",
    "batch_size": 16,
    "learning_rate": 5e-5,
    "epochs": 3,
    "device": "cuda"
}

with open("train_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

print("配置已保存到 train_config.json")
```

### 场景 2：用 Pandas 清洗后保存为 CSV

```python
import pandas as pd

# 模拟清洗后的数据
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [85.5, 90.0, 78.5]
})

# 保存为 CSV（不带索引）
df.to_csv("students.csv", index=False, encoding="utf-8")

# 再读回来验证
df_loaded = pd.read_csv("students.csv", encoding="utf-8")
print(df_loaded.dtypes)
```

### 场景 3：CSV 与 JSON 互转验证

```python
# 1. 将上面的 students.csv 转为 JSON
csv_to_json("students.csv", "students.json")

# 2. 查看 JSON 内容
with open("students.json", "r", encoding="utf-8") as f:
    print(f.read())

# 3. 再转回 CSV
json_to_csv("students.json", "students_roundtrip.csv")

# 4. 用 Pandas 对比两个 CSV 是否一致
df1 = pd.read_csv("students.csv")
df2 = pd.read_csv("students_roundtrip.csv")
print(df1.equals(df2))  # 应输出 True
```

## 七、常见问题与解决（①+④）

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| JSON 中中文显示为 `\uXXXX` | `ensure_ascii` 默认为 `True` | `json.dumps(..., ensure_ascii=False)` |
| CSV 写入后有多余空行 | Windows 下未指定 `newline=''` | `open(..., newline='')` |
| CSV 读取后数字变成字符串 | `csv.reader` 返回的都是字符串 | 用 `int()` / `float()` 转换，或直接用 Pandas |
| `DictWriter` 缺少表头 | 忘记调用 `writeheader()` | 在 `writerows()` 前加 `writer.writeheader()` |
| JSON 转 CSV 报错 | JSON 是嵌套结构，CSV 无法表示 | 先扁平化嵌套数据，或只提取需要的字段 |
| `json.loads` 报错 | JSON 字符串用了单引号 | JSON 标准必须用双引号，修正源数据 |

## 八、总结与后续学习路线（①+②+④）

恭喜你完成了格式转换章节的学习！你现在应该掌握了：

- ✅ JSON 与 Python 类型的对应关系
- ✅ `json.dumps` / `json.loads` / `json.dump` / `json.load` 的区别与用法
- ✅ `ensure_ascii=False` 和 `indent` 参数的作用
- ✅ CSV 的列表读写（`csv.reader` / `csv.writer`）和字典读写（`DictReader` / `DictWriter`）
- ✅ `newline=''` 和 `writeheader()` 的重要性
- ✅ Pandas 的 `read_csv` / `to_csv` / `read_json` / `to_json` 便捷用法
- ✅ JSON 与 CSV 的互转方法

> **下一步预告**：在下一节 **"NumPy 进行基本数组运算"** 中，我们将学习 NumPy 数组的创建、索引、切片和基本运算。届时我们会用 `np.loadtxt()` 或 Pandas 从 CSV 加载数据，转换为 NumPy 数组进行处理。随后还将进入 **"正则表达式进行文本预处理"**，学习如何用 `re` 模块从非结构化文本中提取结构化信息。现在，试着把上节清洗后的数据保存为 CSV 和 JSON，再互相转换验证一下吧！
