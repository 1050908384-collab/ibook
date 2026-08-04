# Python 数据结构：列表、元组、集合与字典精讲

> **参考文章链接**
> 
> ① 底稿：《Python 学习笔记（第五期）——组合数据类型：列表、元组、集合与字典精讲》
> 　　网址：https://blog.csdn.net/AOwhisky/article/details/162815187
> 
> ② 参考文章1：《Python基础篇（三）-- 列表、元组、字典、集合、字符串》
> 　　网址：https://blog.csdn.net/xq151750111/article/details/115336604

---

**适合人群**：已掌握 Python 基础语法（变量、运算符、条件循环），需要系统学习如何组织和操作批量数据的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够根据场景选择合适的数据结构，熟练进行增删改查操作，为后续"函数与面向对象""数据处理"打下坚实基础。

**预计学习时长**：约 2 小时

---

## 第一章：认识组合数据类型——数据的"收纳艺术"

### 1.1 为什么要学组合数据类型

前面的章节里，你学会了用变量存储单个数据：`age = 18`、`name = "Alice"`。但现实中的数据往往是成批出现的——一个班级有 50 个学生成绩、一份通讯录有几百个联系人、一段文本有成千上万个字符。

**组合数据类型**的作用，就是把多个数据组织成一个整体，让程序能够高效地管理和操作。Python 提供了四种核心组合数据类型：

| 数据类型 | 符号 | 可变性 | 有序性 | 元素唯一性 | 核心特征 |
|----------|------|--------|--------|------------|----------|
| **列表（list）** | `[]` | ✅ 可变 | ✅ 有序 | ❌ 允许重复 | 最灵活，随意增删改 |
| **元组（tuple）** | `()` | ❌ 不可变 | ✅ 有序 | ❌ 允许重复 | 数据安全，不可篡改 |
| **集合（set）** | `{}` | ✅ 可变 | ❌ 无序 | ✅ 必须唯一 | 自动去重，集合运算 |
| **字典（dict）** | `{key:value}` | ✅ 可变 | ✅ 保持插入顺序* | **键** 必须唯一 | 键值对映射，极速查找 |

> \* **注意**：从 **Python 3.7 起，`dict` 正式保证按"插入顺序"保留键值对**（CPython 3.6 已作为实现细节存在）。所以字典**不再是"无序"的**——遍历时元素会按插入先后出现。但要区分：dict 的"有序"只是**保留插入顺序**，并不像列表那样支持按整数下标索引，也不会自动排序。相比之下，**集合（set）确实是无序的**。

**为什么这样分类**：根据数据组织方式，Python 将组合类型分为三大类：
- **序列类型**（列表、元组、字符串）：元素按顺序排列，支持索引和切片
- **集合类型**（集合）：元素无序、不重复
- **映射类型**（字典）：通过"键"找"值"，类似查字典

---

### 1.2 序列的通用特征

列表、元组、字符串都属于**序列**。它们共享一些通用操作：

**索引**：每个元素都有一个编号，从 0 开始（正向），或从 -1 开始（反向）。

```
字符串 "Python" 的索引：

正向索引：   0    1    2    3    4    5
           ┌────┬────┬────┬────┬────┬────┐
           │ P  │ y  │ t  │ h  │ o  │ n  │
           └────┴────┴────┴────┴────┴────┘
反向索引：  -6   -5   -4   -3   -2   -1
```

```python
text = "Python"
print(text[0])     # P（第一个字符）
print(text[-1])    # n（最后一个字符）
```

**切片**：截取序列的一部分，语法为 `序列[头下标:尾下标:步长]`，遵循**左闭右开**原则（包含开始位置，不包含结束位置）。

```python
text = "Python"

print(text[1:4])   # yth（索引1到3）
print(text[:3])    # Pyt（从头开始到索引2）
print(text[3:])    # hon（从索引3到末尾）
print(text[::2])   # Pto（从头到尾，步长为2）
print(text[::-1])  # nohtyP（反转序列）
```

**为什么这样写**：`[::-1]` 是反转序列最 Pythonic 的写法，不需要手写循环。

**序列相加与相乘**：

```python
# + 拼接（类型必须相同）
print([1, 2] + [3, 4])      # [1, 2, 3, 4]
print("Hello" + "World")    # HelloWorld

# * 重复
print([1, 2] * 3)           # [1, 2, 1, 2, 1, 2]
print("-" * 30)             # ------------------------------
```

**成员判断**：

```python
print(3 in [1, 2, 3])           # True
print("a" not in "hello")       # True
```

**序列内置函数**：

```python
numbers = [3, 1, 4, 1, 5]

print(len(numbers))      # 5，元素个数
print(max(numbers))      # 5，最大值
print(min(numbers))      # 1，最小值
print(sum(numbers))      # 14，求和（元素必须是数字）
print(sorted(numbers))   # [1, 1, 3, 4, 5]，排序后返回新列表
print(list(reversed(numbers)))  # [5, 1, 4, 1, 3]，反转

# enumerate：同时获取索引和元素
for index, value in enumerate(numbers):
    print(f"索引{index}: {value}")
```

---

## 第二章：列表（List）——灵活的数据"购物车"

### 2.1 创建列表

列表是 Python 中最常用的数据结构，用中括号 `[]` 包裹，元素之间用逗号分隔。

```python
# 空列表
empty = []

# 同类型元素
scores = [85, 92, 78, 96, 88]

# 混合类型（允许，但不推荐）
mixed = [1, "hello", 3.14, True]

# 从其他类型转换
chars = list("Python")          # ['P', 'y', 't', 'h', 'o', 'n']
numbers = list(range(5))       # [0, 1, 2, 3, 4]
```

**为什么这样写**：`list()` 函数可以将字符串、元组、range 对象等可迭代对象转换为列表，非常实用。

### 2.2 访问列表元素

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]

# 索引访问
print(fruits[0])     # 苹果
print(fruits[-1])    # 西瓜（最后一个）

# 切片访问（返回新列表）
print(fruits[1:4])   # ['香蕉', '橙子', '葡萄']
print(fruits[::2])   # ['苹果', '橙子', '西瓜']（步长2）
print(fruits[::-1])  # ['西瓜', '葡萄', '橙子', '香蕉', '苹果']（反转）
```

### 2.3 添加元素

| 方法 | 作用 | 示例 | 结果 |
|------|------|------|------|
| `append(x)` | 末尾添加**单个元素** | `[1,2].append(3)` | `[1, 2, 3]` |
| `extend(iterable)` | 末尾添加**多个元素**（逐个添加） | `[1,2].extend([3,4])` | `[1, 2, 3, 4]` |
| `insert(i, x)` | 在指定位置插入 | `[1,2,3].insert(1, 99)` | `[1, 99, 2, 3]` |

```python
numbers = [1, 2, 3]

numbers.append(4)
print(numbers)        # [1, 2, 3, 4]

numbers.extend([5, 6])
print(numbers)        # [1, 2, 3, 4, 5, 6]

numbers.insert(0, 0)  # 在索引0处插入0
print(numbers)        # [0, 1, 2, 3, 4, 5, 6]
```

**⚠️ 关键区别**：`append()` 和 `extend()` 极易混淆！

```python
lst = [1, 2]

# append 把列表当做一个整体添加
lst.append([3, 4])
print(lst)   # [1, 2, [3, 4]]  ← 注意，[3,4] 是一个元素！

lst2 = [1, 2]

# extend 把列表拆开，逐个添加
lst2.extend([3, 4])
print(lst2)  # [1, 2, 3, 4]
```

**为什么这样写**：`append` 适合添加单个元素，`extend` 适合合并两个列表。用错了会导致嵌套列表，后续遍历时会出问题。

### 2.4 删除元素

| 方法 | 作用 | 特点 |
|------|------|------|
| `del list[i]` | 删除指定索引的元素 | 按位置删，不返回值 |
| `pop(i)` | 删除指定索引的元素 | 默认删最后一个，**返回被删的值** |
| `remove(x)` | 删除第一个值为 x 的元素 | 按值删，找不到会报错 |
| `clear()` | 清空所有元素 | 列表变为 `[]` |

```python
fruits = ["苹果", "香蕉", "橙子", "香蕉"]

del fruits[0]
print(fruits)         # ['香蕉', '橙子', '香蕉']

fruits.remove("香蕉")  # 只删第一个匹配
print(fruits)         # ['橙子', '香蕉']

last = fruits.pop()   # 弹出最后一个
print(last)           # 香蕉
print(fruits)         # ['橙子']

fruits.clear()
print(fruits)         # []
```

### 2.5 修改与查找

```python
numbers = [3, 1, 4, 1, 5]

# 修改单个元素
numbers[0] = 99
print(numbers)   # [99, 1, 4, 1, 5]

# 修改一组元素（切片赋值）
numbers[1:3] = [7, 8]
print(numbers)   # [99, 7, 8, 1, 5]

# 查找元素位置
print(numbers.index(8))    # 2（8的索引）
print(numbers.count(1))    # 1（1出现的次数）
```

### 2.6 列表排序

| 方法 | 作用 | 是否原地修改 | 返回值 |
|------|------|-------------|--------|
| `list.sort()` | 升序排列 | ✅ 是 | `None` |
| `list.sort(reverse=True)` | 降序排列 | ✅ 是 | `None` |
| `sorted(list)` | 升序排列 | ❌ 否 | **新列表** |
| `list.reverse()` | 反转 | ✅ 是 | `None` |

```python
nums = [6, 2, 5, 3, 1, 4]

# sort：原地排序，原列表改变
nums.sort()
print(nums)              # [1, 2, 3, 4, 5, 6]

# sorted：返回新列表，原列表不变
original = [3, 1, 4, 1, 5]
new_list = sorted(original)
print(original)          # [3, 1, 4, 1, 5]（不变）
print(new_list)          # [1, 1, 3, 4, 5]

# 按字符串长度排序
words = ["apple", "fig", "banana", "kiwi"]
words.sort(key=len)
print(words)             # ['fig', 'kiwi', 'apple', 'banana']
```

**为什么这样写**：如果你需要保留原列表，用 `sorted()`；如果不需要保留原顺序，用 `list.sort()` 更省内存。

### 2.7 列表推导式——一行代码的艺术

列表推导式是 Python 的招牌特性，用一行代码替代传统的 `for` 循环。

```python
# 传统写法：生成 1~10 的平方
squares = []
for i in range(1, 11):
    squares.append(i ** 2)

# 列表推导式（一行搞定）
squares = [i ** 2 for i in range(1, 11)]
print(squares)   # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

**带条件的推导式**：

```python
# 1~20 中的偶数
evens = [i for i in range(1, 21) if i % 2 == 0]
print(evens)   # [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# 嵌套循环
list1 = [1, 2, 3]
list2 = [10, 20]
result = [x + y for x in list1 for y in list2]
print(result)  # [11, 21, 12, 22, 13, 23]
```

**语法结构**：`[表达式 for 变量 in 可迭代对象 if 条件]`

---

## 第三章：元组（Tuple）——不可变的"密封档案"

元组和列表几乎一模一样，但有一个核心区别：**元组一旦创建，就不能修改**。

### 3.1 创建元组

```python
# 正常元组
t = (1, 2, 3, "hello")

# ⚠️ 单元素元组必须加逗号！
single = (5,)      # 这是元组
not_tuple = (5)   # 这是整数 5！

# 用 tuple() 转换
t2 = tuple([1, 2, 3])     # (1, 2, 3)
t3 = tuple("Python")      # ('P', 'y', 't', 'h', 'o', 'n')
```

**为什么这样写**：`(5)` 会被 Python 解释为数学括号，结果是整数 5。加逗号 `(5,)` 才是元组的语法标记。

### 3.2 访问元组元素

访问方式与列表完全相同——支持索引、切片和循环遍历。

```python
t = ("a", "b", "c", "d", "e")

print(t[0])        # a
print(t[-1])       # e
print(t[1:4])      # ('b', 'c', 'd')

for item in t:
    print(item, end=" ")   # a b c d e
```

### 3.3 元组的不可变性

```python
t = (1, 2, 3)

# ❌ 以下操作全部报错
# t[0] = 100       # TypeError
# t.append(4)      # AttributeError
# t.remove(2)      # AttributeError
```

**为什么需要不可变**：元组比列表更快、更安全。它适合存储固定不变的数据，如坐标 `(x, y)`、RGB 颜色值 `(255, 128, 0)`。更重要的是，**元组可以作为字典的键**，而列表不行。

### 3.4 列表 vs 元组

| 对比维度 | 列表（List） | 元组（Tuple） |
|----------|-------------|---------------|
| 是否可变 | ✅ 可变 | ❌ 不可变 |
| 是否有序 | ✅ 有序 | ✅ 有序 |
| 性能 | 较慢 | **更快** |
| 作为字典键 | ❌ 不允许 | ✅ 允许 |
| 使用场景 | 动态数据集合 | 固定数据、配置项 |

---

## 第四章：集合（Set）——唯一无序的"去重收纳盒"

### 4.1 创建集合

```python
# 使用大括号（注意：{} 是空字典，不是空集合！）
s = {1, 2, 3, 3, 3}

# 自动去重
print(s)   # {1, 2, 3}

# 空集合必须用 set()
empty = set()
print(type(empty))   # <class 'set'>

# 从列表转换（常用于去重）
unique = set([1, 2, 2, 3, 3, 4])
print(unique)   # {1, 2, 3, 4}
```

**为什么这样写**：`{}` 在 Python 中被优先解释为字典。要创建空集合，只能用 `set()`。

### 4.2 添加与删除

```python
s = {1, 2, 3}

# 添加元素
s.add(4)
print(s)          # {1, 2, 3, 4}

s.add(2)          # 已存在，不做任何事
print(s)          # {1, 2, 3, 4}

# 删除元素
s.remove(1)       # 元素不存在会报错 KeyError
s.discard(99)     # 元素不存在不报错，安全删除
popped = s.pop()  # 随机弹出一个元素
```

### 4.3 集合运算

集合最擅长做数学集合运算：

```python
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# 并集：所有元素
print(A | B)            # {1, 2, 3, 4, 5, 6, 7, 8}
print(A.union(B))

# 交集：共同元素
print(A & B)            # {4, 5}
print(A.intersection(B))

# 差集：在 A 中但不在 B 中
print(A - B)            # {1, 2, 3}
print(A.difference(B))

# 对称差集：只在 A 或只在 B 中
print(A ^ B)            # {1, 2, 3, 6, 7, 8}
print(A.symmetric_difference(B))
```

**为什么这样写**：集合基于哈希表实现，**单个元素的成员判断（`x in s`）平均时间复杂度是 O(1)**，因此在"去重""判断是否存在"等场景下极其高效。但要注意：**并集、交集、差集这类集合运算并不是 O(1)**——它们需要遍历元素，时间复杂度大致与参与运算的集合大小成正比（约 O(len(s)) 或 O(len(s)+len(t))），只是常数因子小、实际速度很快而已。

### 4.4 集合推导式

```python
# 1~10 中偶数的集合
evens = {i for i in range(1, 11) if i % 2 == 0}
print(evens)   # {2, 4, 6, 8, 10}
```

---

## 第五章：字典（Dict）——键值对的"新华字典"

字典是 Python 中唯一的**映射类型**，通过"键"（key）快速查找"值"（value），查找速度极快。

### 5.1 创建字典

```python
# 方式1：大括号
person = {"name": "张三", "age": 18, "city": "北京"}

# 方式2：dict() 函数（键名不加引号）
p2 = dict(name="李四", age=20)

# 方式3：从键值对列表创建
p3 = dict([("name", "王五"), ("age", 22)])

# 方式4：zip 合并两个列表
keys = ["a", "b", "c"]
values = [97, 98, 99]
p4 = dict(zip(keys, values))
print(p4)   # {'a': 97, 'b': 98, 'c': 99}

# 方式5：fromkeys 设置默认值
scores = dict.fromkeys(["语文", "数学", "英语"], 0)
print(scores)   # {'语文': 0, '数学': 0, '英语': 0}
```

**为什么这样写**：字典的键必须是**不可变类型**（字符串、数字、元组），且必须唯一。值可以是任意类型。

### 5.2 访问字典元素

```python
person = {"name": "张三", "age": 18, "city": "北京"}

# 方式1：通过键访问（键不存在会报错 KeyError）
print(person["name"])     # 张三
# print(person["score"])  # KeyError!

# 方式2：get() 方法（推荐，键不存在不报错）
print(person.get("name"))        # 张三
print(person.get("score"))       # None
print(person.get("score", 0))    # 0（指定默认值）

# 方式3：判断键是否存在
print("name" in person)          # True
print("score" not in person)     # True
```

**为什么这样写**：`person["xxx"]` 在键不存在时直接报错，可能导致程序崩溃。`get()` 方法更健壮，适合不确定键是否存在的场景。

### 5.3 添加与修改

```python
person = {"name": "张三", "age": 18}

# 添加新键值对（键不存在）
person["city"] = "北京"
print(person)   # {'name': '张三', 'age': 18, 'city': '北京'}

# 修改已有键的值（键已存在）
person["age"] = 19
print(person)   # {'name': '张三', 'age': 19, 'city': '北京'}

# 批量更新
person.update({"age": 20, "job": "工程师"})
print(person)   # {'name': '张三', 'age': 20, 'city': '北京', 'job': '工程师'}
```

### 5.4 删除元素

```python
person = {"name": "张三", "age": 18, "city": "北京", "job": "工程师"}

# pop(key)：删除指定键，返回对应的值
removed = person.pop("city")
print(removed)      # 北京
print(person)       # {'name': '张三', 'age': 18, 'job': '工程师'}

# popitem()：删除并返回最后插入的键值对（Python 3.7+）
last = person.popitem()
print(last)         # ('job', '工程师')

# clear()：清空
person.clear()
print(person)       # {}
```

### 5.5 遍历字典

```python
person = {"name": "张三", "age": 18, "city": "北京"}

# 遍历键（默认）
for key in person:
    print(key, ":", person[key])

# 遍历键值对（最常用）
for key, value in person.items():
    print(f"{key} = {value}")

# 单独遍历键或值
print(list(person.keys()))      # ['name', 'age', 'city']
print(list(person.values()))    # ['张三', 18, '北京']
```

### 5.6 字典推导式

```python
# 交换键和值
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
print(swapped)   # {1: 'a', 2: 'b', 3: 'c'}

# 过滤：只保留值大于1的项
filtered = {k: v for k, v in original.items() if v > 1}
print(filtered)  # {'b': 2, 'c': 3}
```

---

## 第六章：组合数据类型通用运算符

| 运算符 | 适用类型 | 作用 | 示例 |
|--------|----------|------|------|
| `+` | 字符串、列表、元组 | 拼接 | `[1,2] + [3,4]` → `[1,2,3,4]` |
| `*` | 字符串、列表、元组 | 重复 | `[1,2] * 3` → `[1,2,1,2,1,2]` |
| `in` | 所有序列 + 集合 + 字典（键） | 判断是否存在 | `"a" in ["a","b"]` → `True` |
| `not in` | 同上 | 判断是否不存在 | `"c" not in ["a","b"]` → `True` |

```python
# 字典的 in 判断的是键，不是值
print("name" in {"name": "张三"})   # True
print("张三" in {"name": "张三"})   # False
```

---

## 第七章：综合实战

### 实战一：十大歌手评分系统

**需求**：10 位评委打分，去掉一个最高分和一个最低分，计算平均分。

```python
scores = [9.5, 8.8, 9.2, 7.5, 9.8, 9.0, 8.5, 9.3, 8.9, 9.1]

print(f"原始打分：{scores}")

# 排序
scores.sort()
print(f"排序后：  {scores}")

# 去掉最高和最低（切片）
valid = scores[1:-1]   # 去掉第一个和最后一个
print(f"有效分数：{valid}")

# 计算平均分
average = sum(valid) / len(valid)
print(f"最终得分：{average:.2f} 分")
```

**输出**：
```
原始打分：[9.5, 8.8, 9.2, 7.5, 9.8, 9.0, 8.5, 9.3, 8.9, 9.1]
排序后：  [7.5, 8.5, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.5, 9.8]
有效分数：[8.5, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.5]
最终得分：9.04 分
```

### 实战二：手机通讯录管理系统

```python
contacts = {}

def show_menu():
    print("\n" + "=" * 30)
    print("      📱 手机通讯录")
    print("=" * 30)
    print("1. 添加联系人")
    print("2. 查看联系人")
    print("3. 修改联系人")
    print("4. 删除联系人")
    print("5. 显示全部")
    print("0. 退出")
    print("=" * 30)

while True:
    show_menu()
    choice = input("请选择功能（0-5）：")

    if choice == "0":
        print("👋 再见！")
        break

    elif choice == "1":
        name = input("请输入姓名：")
        if name in contacts:
            print(f"⚠️ {name} 已存在！")
            continue
        phone = input("请输入电话：")
        contacts[name] = phone
        print(f"✅ 已添加：{name} -> {phone}")

    elif choice == "2":
        name = input("请输入要查询的姓名：")
        phone = contacts.get(name)
        if phone:
            print(f"📞 {name} 的电话：{phone}")
        else:
            print(f"❌ 未找到：{name}")

    elif choice == "3":
        name = input("请输入要修改的姓名：")
        if name in contacts:
            new_phone = input(f"请输入 {name} 的新电话：")
            contacts[name] = new_phone
            print(f"✅ 已更新")
        else:
            print(f"❌ 未找到：{name}")

    elif choice == "4":
        name = input("请输入要删除的姓名：")
        if name in contacts:
            confirm = input(f"确认删除 {name}？（y/n）：")
            if confirm.lower() == "y":
                del contacts[name]
                print(f"✅ 已删除")
        else:
            print(f"❌ 未找到：{name}")

    elif choice == "5":
        if not contacts:
            print("📭 通讯录为空")
        else:
            print("\n📋 全部联系人：")
            for name, phone in contacts.items():
                print(f"  {name} : {phone}")
            print(f"共 {len(contacts)} 位")

    else:
        print("❌ 无效输入")
```

### 实战三：找出两个班级的共同学生

```python
class_a = {"张三", "李四", "王五", "赵六"}
class_b = {"李四", "赵六", "孙七", "周八"}

# 共同学生（交集）
common = class_a & class_b
print(f"两个班都有的学生：{common}")

# 只在 A 班的学生（差集）
only_a = class_a - class_b
print(f"只在 A 班的学生：{only_a}")

# 所有学生（并集，自动去重）
all_students = class_a | class_b
print(f"所有学生：{all_students}")
```

---

## 第八章：常见错误与避坑清单

| 错误现象 | 原因 | 解决方案 |
|----------|------|----------|
| `TypeError: unhashable type: 'list'` | 用列表作为字典的键或集合的元素 | 改用元组（不可变类型） |
| `KeyError: 'xxx'` | 访问字典中不存在的键 | 使用 `dict.get(key, default)` |
| `AttributeError: 'tuple' object has no attribute 'append'` | 误以为元组可以修改 | 确认是否需要用列表替代 |
| 遍历时删除列表元素结果异常 | 遍历时修改了列表长度 | 遍历副本：`for item in list[:]` |
| `append()` 后列表出现嵌套 | 把列表 append 到列表 | 应该用 `extend()` |
| `{}` 创建的不是空集合 | `{}` 被解释为字典 | 空集合用 `set()` |
| `(5)` 不是元组 | 单元素没加逗号 | 写成 `(5,)` |

---

## 本章小结

| 数据类型 | 创建方式 | 核心操作 | 适用场景 |
|----------|----------|----------|----------|
| **列表** | `[1, 2, 3]` | append/extend/insert/pop/remove/sort | 动态数据、需要频繁增删改 |
| **元组** | `(1, 2, 3)` | 索引、切片（只读） | 固定数据、作为字典键 |
| **集合** | `{1, 2, 3}` | add/remove/并交差运算 | 去重、集合运算 |
| **字典** | `{"a": 1}` | get/items/keys/values/pop/update | 键值对映射、快速查找 |

**养成的好习惯**：
1. 需要动态增删改 → 用**列表**
2. 数据固定不变 → 用**元组**
3. 需要去重 → 用**集合**
4. 需要通过名称查找值 → 用**字典**
5. 访问字典优先用 `get()`，避免 `KeyError`
6. 能用列表推导式的地方，尽量不用 `for` 循环

**下一步预告**：掌握了数据结构后，你将进入 **函数与面向对象** 的学习——学会把重复代码封装成可复用的函数，理解参数传递、返回值、作用域等核心概念，让你的代码从"能跑"走向"优雅"。继续加油！