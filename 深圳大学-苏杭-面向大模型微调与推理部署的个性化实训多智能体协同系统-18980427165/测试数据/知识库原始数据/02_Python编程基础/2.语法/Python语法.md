# Python 语法：从零写出第一行代码

> **参考文章链接**
> 
> ① 底稿：《Python语法》
> 　　网址：https://blog.csdn.net/HHX_01/article/details/160450852
> 
> ② 参考文章：《Python 3 基础语法详解：从入门到实践》
> 　　网址：https://blog.csdn.net/a8039974/article/details/161197007
> 
> ③ 参考文章：《Python - Python基础：注释；基本数据类型；变量；基本运算符；判断循环流程（二）》
> 　　网址：https://blog.csdn.net/MinggeQingchun/article/details/143760253

---

**适合人群**：已完成环境准备、初次接触 Python 编程的零基础学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够独立编写结构清晰、逻辑正确的 Python 程序，为后续"数据结构""函数与面向对象"等进阶内容打下坚实基础。

**预计学习时长**：约 2 小时  
**环境准备**：Python 3.10+，任意代码编辑器（VS Code / PyCharm / 甚至记事本都可以）

---

## 第一章：代码书写规范——Python 的"交通规则"

在开车上路前，你需要知道红绿灯和车道规则。写 Python 也一样，先掌握最基础的代码书写规范，后面才不会频繁报错。

### 1.1 缩进：Python 的灵魂

**绝大多数编程语言**（如 C、Java）用大括号 `{}` 包裹代码块，而 **Python 强制使用缩进** 来划分代码层级。这是 Python 最显著的特征，也是新手最容易踩的坑。

**写法**：统一使用 **4 个空格** 作为一级缩进。

```python
if 10 > 5:
    print("条件成立")      # 前面有4个空格，属于 if 代码块内部
    print("这行也在 if 里") # 同样缩进4个空格
print("代码执行结束")        # 无缩进，不属于 if 代码块
```

**为什么这样写**：Python 解释器通过缩进判断哪些代码属于同一个逻辑块。如果同一代码块内的空格数不一致，程序会直接抛出 `IndentationError`（缩进错误）并停止运行。

**⚠️ 避坑提醒**：
- 不要混用 Tab 键和空格，选一个就坚持到底（推荐空格）。
- 不要写 `if 10 > 5: print("成立")` 这种挤在一行的写法，降低可读性。

---

### 1.2 注释：写给人类看的文字

注释不会被程序执行，它的作用是解释代码逻辑，方便日后回顾或他人阅读。

**单行注释**：以 `#` 开头，`#` 后面的内容都会被忽略。

```python
# 这是单行注释，解释下面这行代码的作用
name = "Python学习者"  # 注释也可以写在代码末尾
```

**多行说明怎么写**：Python **没有真正的"多行注释"语法**。要注释掉多行代码，规范做法是**每行前面各写一个 `#`**（编辑器一般有"批量注释"快捷键，如 VS Code 的 `Ctrl + /`）。

```python
# 这几行是注释，用来说明下面的逻辑
# 每一行都以 # 开头
# 这才是 Python 规范的多行注释写法
name = "Python学习者"
```

**关于三引号 `'''...'''` / `"""..."""`**：它们包裹的其实是一个**字符串字面量**（多行字符串），并不是注释——解释器会真正创建这个字符串对象。只是当它**单独成一条语句、其返回值没有被使用**时，看起来"像"注释而已。因此**不要把三引号当多行注释来用**：写在函数、循环等语句中间反而可能改变代码含义。它真正的规范用途是 **docstring（文档字符串）**。

```python
'''
这是一个多行字符串字面量。
只有当它单独出现、且作为 docstring 放在模块/函数/类的开头时才有意义，
不要拿它来临时"注释掉"代码。
'''
```

**文档注释（docstring）**：放在**模块、函数或类定义的第一行**，用 `"""` 包裹，用于描述其功能，可通过 `help()` 或 `__doc__` 读取，也是生成项目文档的标准写法。后续学习函数时会详细用到。

---

### 1.3 语句结束与换行

**为什么这样写**：Python 不需要像 C/Java 那样在语句末尾加分号 `;`。正常情况下，**一行就是一条语句**，直接换行即可。

```python
a = 10
b = 20
print(a + b)
```

如果一行确实写不下，可以用反斜杠 `\` 续行，但更好的做法是用括号 `()` 自动换行：

```python
# 不推荐：反斜杠续行
total = item_one + \
        item_two + \
        item_three

# 推荐：括号内自动换行，不需要反斜杠
total = (item_one +
         item_two +
         item_three)
```

---

### 1.4 标识符命名规范

标识符就是你给变量、函数、类起的名字。Python 对命名有**硬性规则**和**行业约定**：

**硬性规则（违反就报错）**：
1. 只能由字母、数字、下划线 `_` 组成。
2. 不能以数字开头。
3. 不能使用 Python 关键字（如 `if`、`for`、`class`）。
4. 严格区分大小写：`Name` 和 `name` 是两个不同的变量。

**行业约定（不遵守不报错，但会被同事嫌弃）**：
- 普通变量/函数名：全小写 + 下划线，如 `user_name`、`student_age`。
- 类名：大驼峰式，如 `MyClass`。
- 常量：全大写 + 下划线，如 `MAX_COUNT`、`PI`。
- 不要用拼音或无意义的单字母（如 `a`、`b`、`x1`）。

**查看所有关键字**：

```python
import keyword
print(keyword.kwlist)
```

---

## 第二章：变量与常量——数据的"收纳盒"

### 2.1 变量是什么

通俗地说，变量就是一个**带名字的收纳盒**。你把数据放进去，之后通过名字就能取出数据。

**写法**：`变量名 = 数据值`。Python 是动态类型语言，不需要提前声明类型，解释器会自动识别。

```python
age = 18                # 整数
score = 95.5            # 浮点数
name = "编程小白"        # 字符串
is_study = True         # 布尔值
```

**为什么这样写**：`=` 在 Python 中不是数学上的"等于"，而是"赋值"——把右边的数据放进左边的盒子里。

### 2.2 多重赋值与链式赋值

```python
# 多重赋值：同时给多个变量赋值
a, b, c = 10, 20, 30

# 链式赋值：多个变量指向同一个值
x = y = z = 100
```

### 2.3 常量

常量指程序运行中**不应被修改**的数据。Python 没有严格的"只读常量"机制，行业约定：**全大写的变量视为常量**，大家自觉不去修改它。

```python
PI = 3.1415926
MAX_RETRY = 3
```

---

## 第三章：基本数据类型——认识你的"原材料"

Python 内置了六大核心数据类型，掌握它们是编写任何程序的基础。

### 3.1 数字（Number）

| 类型 | 说明 | 示例 |
|------|------|------|
| `int` | 整数 | `100`, `-5`, `0` |
| `float` | 浮点数 | `3.14`, `-0.5`, `2.5e2` |
| `bool` | 布尔值 | `True`（真，等价于1）、`False`（假，等价于0） |
| `complex` | 复数 | `3 + 4j` |

```python
num1 = 666      # int
num2 = 88.9     # float
flag = True     # bool，可以参与数学运算：True + 1 结果是 2
```

### 3.2 字符串（str）

字符串用于存储文本，用单引号、双引号或三引号包裹。

```python
s1 = 'Hello'
s2 = "Python"
s3 = '''多行
字符串'''
```

**索引与切片**：字符串是有序序列，每个字符都有编号（从 0 开始）。

```python
text = "Python"
print(text[0])      # P，第一个字符
print(text[-1])     # n，最后一个字符
print(text[0:4])    # Pyth，切片：左闭右开，取索引 0~3
print(text[::2])    # Pto，从头到尾，步长为2
```

**为什么这样写**：`[start:end:step]` 是 Python 中非常高效的子串截取方式，不需要手写循环。

**常用方法**：

```python
text = "  Hello Python  "
print(text.strip())           # 去除首尾空格："Hello Python"
print(text.upper())           # 全转大写
print(text.replace("Python", "World"))  # 替换子串
print(text.split())           # 按空格分割为列表：['Hello', 'Python']
print(len(text))              # 获取长度
```

**字符串格式化**（三种方式）：

```python
name = "Alice"
age = 20

# 方式1：% 格式化（较老，了解即可）
print("我叫%s，今年%d岁" % (name, age))

# 方式2：format() 方法
print("我叫{}，今年{}岁".format(name, age))

# 方式3：f-string（推荐，Python 3.6+）
print(f"我叫{name}，今年{age}岁")
```

### 3.3 列表（list）

列表是一个**有序、可变**的集合，用中括号 `[]` 定义，元素类型可以不同。

```python
my_list = [10, "hello", 3.14, True]
```

**基本操作**：

```python
my_list = [1, 2, 3]

# 增
my_list.append(4)        # 末尾追加：[1, 2, 3, 4]
my_list.insert(0, 0)   # 指定位置插入：[0, 1, 2, 3, 4]

# 删
my_list.pop()            # 删除并返回最后一个元素
my_list.remove(2)        # 删除第一个值为2的元素

# 改
my_list[0] = 99          # 通过索引修改

# 查
print(my_list[1])        # 通过索引取值
print(len(my_list))      # 获取长度
```

### 3.4 元组（tuple）

元组与列表类似，但**一旦创建就不能修改**，用小括号 `()` 定义。适合存储固定不变的数据。

```python
my_tuple = (1, 2, "fixed")
# my_tuple[0] = 99  # 报错！元组不可变
```

**为什么这样写**：元组的"不可变性"让它比列表更安全，在函数传参、数据保护场景中经常使用。

### 3.5 字典（dict）

字典以**键值对（key-value）**形式存储数据，用大括号 `{}` 定义，查询效率极高。

```python
user = {"name": "小明", "age": 20, "city": "北京"}
```

**基本操作**：

```python
# 查
print(user["name"])           # 小明。键不存在会报错
print(user.get("gender"))     # None。键不存在不会报错，更安全
print(user.keys())            # 所有键
print(user.values())          # 所有值
print(user.items())           # 所有键值对

# 增/改
user["job"] = "工程师"         # 键不存在则新增，存在则修改
user.update({"age": 21, "gender": "男"})  # 批量更新

# 删
del user["city"]              # 删除指定键值对
age = user.pop("age")         # 弹出并返回值
```

**为什么这样写**：字典的键必须是**不可变类型**（如字符串、数字、元组），且必须唯一。它是存储结构化数据（如用户信息、配置参数）的首选。

### 3.6 集合（set）

集合是**无序、不重复**的元素集合，用 `{}` 定义（空集合必须用 `set()`）。

```python
s = {1, 2, 3, 3, 3}
print(s)  # {1, 2, 3}，自动去重
```

**集合运算**：

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)   # 并集：{1, 2, 3, 4, 5, 6}
print(a & b)   # 交集：{3, 4}
print(a - b)   # 差集：{1, 2}
print(a ^ b)   # 对称差集：{1, 2, 5, 6}
```

### 3.7 类型检查与转换

**检查类型**：

```python
print(type(10))              # <class 'int'>
print(isinstance(3.14, float))  # True
```

**类型转换**：

```python
a = "50"
b = int(a)        # 字符串转整数：50
c = str(66.6)     # 数字转字符串："66.6"
d = list("abc")   # 字符串转列表：['a', 'b', 'c']
e = float("3.14") # 字符串转浮点数：3.14
```

**⚠️ 避坑提醒**：`int("3.14")` 会报错，因为字符串 `"3.14"` 包含小数点，无法直接转整数。正确做法是先 `float("3.14")` 再 `int()`。

---

## 第四章：运算符——让数据"动起来"

### 4.1 算术运算符

```python
print(10 + 3)    # 13，加法
print(10 - 3)    # 7，减法
print(10 * 3)    # 30，乘法
print(10 / 3)    # 3.333...，除法（结果一定是浮点数）
print(10 // 3)   # 3，整除（地板除，只保留整数部分）
print(10 % 3)    # 1，取余（模运算）
print(2 ** 3)    # 8，幂运算（2的3次方）
```

### 4.2 赋值运算符

```python
a = 10
a += 5   # 等价于 a = a + 5，a 变为 15
a -= 3   # 等价于 a = a - 3，a 变为 12
a *= 2   # 等价于 a = a * 2，a 变为 24
```

### 4.3 比较运算符

结果永远是布尔值 `True` 或 `False`。

```python
print(20 > 15)    # True
print(10 == 5)    # False，注意是两个等号
print(10 != 5)    # True，不等于
```

### 4.4 逻辑运算符

| 运算符 | 含义 | 规则 |
|--------|------|------|
| `and` | 与 | 两边都为真，结果才为真 |
| `or` | 或 | 只要一边为真，结果就为真 |
| `not` | 非 | 取反 |

```python
print(10 > 5 and 8 < 20)   # True
print(10 > 5 or 8 > 20)    # True
print(not (10 > 5))        # False
```

**短路运算**（进阶理解）：
- `and` 运算中，如果左边为假，右边不会执行，直接返回左边的假值。
- `or` 运算中，如果左边为真，右边不会执行，直接返回左边的真值。

```python
# 验证短路
a = 0
b = 5
print(a and b)   # 0，因为 a 为"假值"，直接返回 a
print(a or b)    # 5，因为 a 为"假值"，返回 b
```

### 4.5 成员运算符与身份运算符

```python
# 成员运算符：判断某个值是否在序列中
print("a" in "abc")        # True
print(4 not in [1, 2, 3])  # True

# 身份运算符：判断是否为同一个对象（内存地址相同）
x = [1, 2, 3]
y = x
z = x[:]  # 创建副本
print(x is y)   # True，指向同一个对象
print(x is z)   # False，z 是新对象
print(x == z)   # True，值相等
```

**为什么这样写**：`==` 比较的是"值"是否相等，`is` 比较的是"身份"（内存地址）是否相同。日常判断值相等用 `==`，判断是否为同一个对象用 `is`。

### 4.6 运算符优先级

不需要死记硬背，记住一条：**有括号先算括号里的**。复杂表达式建议多写括号，提高可读性。

大致优先级（从高到低）：
1. `()` 括号
2. `**` 幂运算
3. `* / // %` 乘除模
4. `+ -` 加减
5. `> < == !=` 比较
6. `not` → `and` → `or` 逻辑运算

---

## 第五章：输入与输出——程序与用户的对话

### 5.1 输出：print()

```python
print("Hello, Python!")
print("a =", 10, "b =", 20)  # 可以同时输出多个内容，自动用空格分隔
```

### 5.2 输入：input()

```python
name = input("请输入你的名字：")
print(f"你好，{name}！")
```

**⚠️ 重要避坑**：`input()` 接收到的内容**永远是字符串类型**！如果需要数字，必须手动转换。

```python
# 错误示范
age = input("请输入年龄：")
print(age + 1)  # 报错！字符串不能和整数相加

# 正确做法
age = int(input("请输入年龄："))
print(f"明年你就{age + 1}岁了")
```

---

## 第六章：程序流程控制——代码的"方向盘"

### 6.1 条件判断：if / elif / else

程序需要根据条件做出选择，Python 用 `if` 系列语句实现。

**单分支**：

```python
score = 85
if score >= 60:
    print("及格了")
```

**双分支**：

```python
age = 16
if age >= 18:
    print("已成年")
else:
    print("未成年")
```

**多分支**：

```python
score = 85
if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")
```

**为什么这样写**：`elif` 是 "else if" 的缩写。程序会**从上到下依次判断**，一旦某个条件满足，执行对应代码块后就会**跳出整个判断结构**，后面的 `elif` 不再检查。

**嵌套 if**：

```python
is_member = True
amount = 200

if is_member:
    if amount >= 100:
        print("会员且满100，打8折")
    else:
        print("会员但未满100，打9折")
else:
    print("非会员，原价")
```

**三元表达式**（简洁写法）：

```python
age = 20
status = "成年" if age >= 18 else "未成年"
print(status)
```

### 6.2 while 循环

适合**不确定循环次数**的场景，只要条件为真就一直执行。

```python
count = 0
while count < 5:
    print(count)
    count += 1   # 每次循环 count 加 1，否则会变成死循环！
```

**while...else**：循环正常结束（没有被 `break` 打断）时执行 `else`。

```python
count = 0
while count < 3:
    print(count)
    count += 1
else:
    print("循环正常结束")
```

### 6.3 for 循环

适合**遍历数据或确定次数**的场景，使用频率比 `while` 更高。

```python
# 遍历字符串
for char in "Python":
    print(char)

# 遍历列表
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# 遍历字典
person = {"name": "Alice", "age": 30}
for key, value in person.items():
    print(f"{key}: {value}")
```

**range() 函数**：生成数字序列，常与 for 配合。

```python
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 7):   # 2, 3, 4, 5, 6（左闭右开）
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8（步长为2）
    print(i)
```

### 6.4 循环控制关键字

| 关键字 | 作用 |
|--------|------|
| `break` | 立即终止整个循环，跳出循环体 |
| `continue` | 跳过当前这一次循环，进入下一次 |
| `pass` | 占位符，不做任何事，保持代码结构完整 |

```python
# break：找到第一个偶数就停止
numbers = [1, 3, 5, 7, 8, 9]
for num in numbers:
    if num % 2 == 0:
        print(f"找到偶数：{num}")
        break

# continue：跳过奇数，只打印偶数
for num in range(10):
    if num % 2 != 0:
        continue
    print(num)  # 输出 0, 2, 4, 6, 8
```

---

## 第七章：综合练习——把知识串起来

请打开你的编辑器，新建一个文件 `practice.py`，完成以下练习。

### 练习 1：简易计算器

```python
# 接收用户输入的两个数字和一个运算符，输出计算结果
num1 = float(input("请输入第一个数字："))
operator = input("请输入运算符（+ - * /）：")
num2 = float(input("请输入第二个数字："))

if operator == "+":
    result = num1 + num2
elif operator == "-":
    result = num1 - num2
elif operator == "*":
    result = num1 * num2
elif operator == "/":
    if num2 == 0:
        result = "错误：除数不能为0"
    else:
        result = num1 / num2
else:
    result = "错误：不支持的运算符"

print(f"计算结果：{result}")
```

### 练习 2：猜数字游戏

```python
import random

target = random.randint(1, 100)  # 生成1~100的随机整数
guess_count = 0

while True:
    guess = int(input("猜一个1到100之间的数字："))
    guess_count += 1
    
    if guess > target:
        print("太大了！")
    elif guess < target:
        print("太小了！")
    else:
        print(f"恭喜猜对了！你一共猜了{guess_count}次")
        break
```

### 练习 3：统计字符串中的元音字母

```python
text = input("请输入一段英文：")
vowels = "aeiouAEIOU"
count = 0

for char in text:
    if char in vowels:
        count += 1

print(f"元音字母共有 {count} 个")
```

---

## 本章小结与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 缩进混用 Tab 和空格 | 统一使用 4 个空格 |
| 用中文标点符号（如 `，` `：`） | 所有代码符号必须是英文半角 |
| `=` 和 `==` 混淆 | `=` 是赋值，`==` 是判断是否相等 |
| `input()` 返回字符串却直接做数学运算 | 先用 `int()` 或 `float()` 转换 |
| 变量名使用关键字（如 `if = 10`） | 避开 `keyword.kwlist` 中的保留字 |
| while 循环忘记写终止条件 | 确保循环变量会趋向终止条件 |
| 字符串直接和数字拼接 | 先用 `str()` 转换，或用 f-string |

---

**下一步预告**：掌握了语法基础后，你将进入 **数据结构** 的深度学习——列表推导式、字典高级操作、元组解包等技巧，让你的代码更 Pythonic。继续加油！