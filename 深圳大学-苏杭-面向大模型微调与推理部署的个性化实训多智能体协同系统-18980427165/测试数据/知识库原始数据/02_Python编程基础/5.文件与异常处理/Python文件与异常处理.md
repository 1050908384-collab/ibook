# 文件与异常处理：让程序既能读写文件，又能优雅地应对意外

> **参考文章链接**
> 
> ① 底稿：《〖python零基础教程第5讲〗文件I/O操作与异常处理》
> 　　网址：https://blog.csdn.net/weixin_44781464/article/details/162623695
> 
> ② 参考文章1：《Python 文件操作与异常处理详解》
> 　　网址：https://blog.csdn.net/zwh821/article/details/156821941
> 
> ③ 参考文章2：《Python 异常处理与文件读写完整指南》
> 　　网址：https://blog.csdn.net/weixin_34396103/article/details/92708967

---

**适合人群**：已掌握 Python 基础语法和数据结构，希望学会读写文件、处理程序运行中意外情况的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够安全地读写文本和二进制文件、处理各种运行时错误，为后续"装饰器与生成器""深度学习框架"打下坚实基础。

**预计学习时长**：约 1.5 小时

---

## 第一章：文件操作基础——打开、读写、关闭

### 1.1 为什么要学文件操作

程序运行时的数据存储在内存中，一旦程序关闭，数据就消失了。**文件操作**让程序能够把数据持久化保存到硬盘上，或者从硬盘读取已有数据。无论是保存训练日志、读取数据集、还是写入配置文件，文件操作都是必备技能。

### 1.2 open() 函数：文件操作的入口

`open()` 是 Python 内置的文件打开函数，它的**完整签名**如下：

```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
```

**核心参数说明**：

| 参数 | 含义 | 说明 |
|------|------|------|
| `file` | 文件路径 | 字符串或 `Path` 对象，必需 |
| `mode` | 打开模式 | 决定读写权限和文件类型，默认 `'r'` |
| `buffering` | 缓冲策略 | `-1` 表示使用系统默认缓冲；`0` 表示不缓冲（仅二进制）；`1` 表示行缓冲（仅文本）；大于1表示指定缓冲区大小 |
| `encoding` | 编码格式 | 文本模式下指定字符编码，**强烈建议显式写 `'utf-8'`** |
| `errors` | 编码错误处理 | 如 `'strict'`（默认，报错）、`'ignore'`（忽略）、`'replace'`（替换） |
| `newline` | 换行符控制 | 控制文本模式下的换行符转换行为 |

**为什么这样写**：`buffering` 参数决定了数据在内存中暂存多少字节后再写入磁盘。默认 `-1` 让系统自动选择，通常性能最优。不指定 `encoding` 时，Python 会使用系统默认编码（Windows 上可能是 `gbk`），跨平台运行时容易出现乱码。显式写 `encoding='utf-8'` 是最佳实践。

### 1.3 文件模式详解

`mode` 参数用一个字符串表示，常见组合如下：

| 模式 | 含义 | 文件不存在时 | 文件存在时 | 指针位置 | 适用场景 |
|------|------|-------------|-----------|----------|----------|
| `'r'` | 只读文本（默认） | 报错 | 保留内容 | 开头 | 读取已有文件 |
| `'w'` | 只写文本 | 创建新文件 | **清空原有内容** | 开头 | 写入新文件 |
| `'a'` | 追加文本 | 创建新文件 | 保留内容 | 末尾 | 写日志 |
| `'r+'` | 读写文本 | 报错 | 保留内容 | 开头 | 读写已有文件 |
| `'w+'` | 读写文本 | 创建新文件 | **清空原有内容** | 开头 | 先写后读新文件 |
| `'a+'` | 读写追加 | 创建新文件 | 保留内容 | 末尾 | 追加读写 |
| `'rb'` | 只读二进制 | 报错 | 保留内容 | 开头 | 读取图片、视频 |
| `'wb'` | 只写二进制 | 创建新文件 | **清空原有内容** | 开头 | 写入图片、视频 |

**关键规则**：文本模式（`t`）可省略，二进制模式（`b`）必须显式写出。

**⚠️ 避坑提醒**：
- `'w'` 和 `'w+'` 模式会**清空原文件**！如果你本意是追加内容，误用了 `'w'`，之前的数据就全没了。
- `'r+'` 模式下文件必须存在，且写入时会从开头覆盖，不会自动追加。
- `'w+'` 打开后会先清空文件，写入后若要读取，需先用 `f.seek(0)` 将光标移回开头。

### 1.4 基础操作流程：打开 → 操作 → 关闭

文件操作必须遵循"打开 → 操作 → 关闭"的流程。**如果不关闭文件**，会导致：
1. 数据可能还在缓冲区，没真正写入磁盘
2. 占用系统资源（文件描述符），累积过多会耗尽
3. 其他程序可能无法访问该文件

**传统写法（不推荐）**：

```python
f = open('test.txt', 'w', encoding='utf-8')
f.write('Hello, Python!')
f.close()   # 如果上面写的时候报错，这行不会执行！
```

**为什么这样写**：如果 `f.write()` 抛出异常，`f.close()` 永远不会执行，文件句柄就泄漏了。在服务器上长期运行的程序，这种泄漏会累积成严重问题。操作系统通过**文件描述符（File Descriptor）**管理打开的文件，Linux 默认每个进程最多打开 1024 个，不释放就会耗尽。

---

## 第二章：with 语句——文件操作的"自动关门器"

### 2.1 为什么必须用 with

`with` 语句是 Python 的**上下文管理器**，它能确保：无论 `with` 代码块内发生什么（正常结束、return、break、抛出异常），文件都会被**自动关闭**。

```python
with open('test.txt', 'w', encoding='utf-8') as f:
    f.write('Hello, Python!')
    # 即使这里抛出异常，文件也会被关闭
# 离开 with 块，文件自动关闭
```

**为什么这样写**：`open()` 返回的文件对象实现了上下文管理协议（`__enter__` 和 `__exit__` 方法）。进入 `with` 时调用 `__enter__`，退出时调用 `__exit__`，而 `__exit__` 里包含了 `close()` 逻辑。这是工程安全的底线。

### 2.2 上下文管理协议的底层原理

`with` 语句的背后是上下文管理协议，需要实现两个特殊方法：

- `__enter__()`：进入上下文时调用，返回资源对象
- `__exit__()`：退出上下文时调用，负责清理资源。它接收异常类型、值、回溯对象三参数，可决定是否"压制"异常（返回 `True`）或继续传播（返回 `False` 或 `None`）

### 2.3 同时操作多个文件

```python
# 复制图片：同时打开源文件和目标文件
with open('source.jpg', 'rb') as src, open('copy.jpg', 'wb') as dst:
    data = src.read()
    dst.write(data)
print('图片复制完成！')
```

**为什么这样写**：`with` 后面可以跟多个 `open()`，用逗号分隔。两个文件都会在 `with` 块结束时自动关闭。

---

## 第三章：文件读取的多种方式

根据文件大小和处理需求，选择不同的读取方式。

### 3.1 read()：一次性读取全部

```python
with open('data.txt', 'r', encoding='utf-8') as f:
    content = f.read()   # 读取全部内容，返回字符串
print(content)
```

**适用场景**：小文件（几 MB 以内）。大文件用 `read()` 会一次性载入内存，可能导致内存不足。

### 3.2 readline()：逐行读取

```python
with open('data.txt', 'r', encoding='utf-8') as f:
    line1 = f.readline()   # 读取第一行
    line2 = f.readline()   # 读取第二行
print(line1, line2)
```

**为什么这样写**：`readline()` 每次只读一行，适合需要按行处理但不需要全部载入内存的场景。注意：返回的字符串**包含末尾的换行符 `\n`**。

### 3.3 readlines()：读取为列表

```python
with open('data.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()   # 返回列表，每个元素是一行
print(lines)   # ['第一行\n', '第二行\n', '第三行']
```

**为什么这样写**：`readlines()` 把整个文件读成一个列表，适合需要按行索引访问的中等大小文件。

### 3.4 最推荐的方式：for 循环逐行迭代

```python
with open('large_log.txt', 'r', encoding='utf-8') as f:
    for line in f:   # 每次只读一行到内存，最省内存
        line = line.strip()   # 去掉首尾空白和换行符
        print(line)
```

**为什么这样写**：这是处理大文件的最佳方式。文件对象本身是可迭代对象，`for line in f` 每次只把一行载入内存，即使文件有几十 GB 也能流畅处理。

### 3.5 写入文件

```python
# 覆盖写入（'w' 模式会清空原文件）
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('第一行内容\n')
    f.write('第二行内容\n')

# 追加写入（'a' 模式保留原有内容）
with open('log.txt', 'a', encoding='utf-8') as f:
    f.write('新的日志记录\n')
```

**为什么这样写**：`write()` 不会自动添加换行符，需要手动写 `\n`。日志场景必须用 `'a'` 模式，否则每次运行都会清空之前的日志。

---

## 第四章：路径操作——找到你的文件

### 4.1 相对路径 vs 绝对路径

- **相对路径**：相对于当前运行目录，如 `'data.txt'`、`'./images/photo.jpg'`
- **绝对路径**：从根目录开始的完整路径，如 `C:\Users\Alice\project\data.txt`

**为什么这样写**：相对路径让项目更便携，但容易因为"当前工作目录不对"而找不到文件。可以用以下代码查看当前工作目录：

```python
import os
print(os.getcwd())   # 查看当前工作目录
```

### 4.2 os.path 模块（传统方式）

```python
import os

# 路径拼接（自动处理不同系统的分隔符 \ 或 /）
path = os.path.join('data', 'subdir', 'file.txt')
print(path)   # Windows: data\subdir\file.txt  Linux/Mac: data/subdir/file.txt

# 获取文件名
print(os.path.basename(path))   # file.txt

# 获取目录名
print(os.path.dirname(path))    # data/subdir

# 获取扩展名
print(os.path.splitext(path))   # ('data/subdir/file', '.txt')

# 判断文件是否存在
print(os.path.exists(path))

# 获取绝对路径
print(os.path.abspath(path))
```

**为什么这样写**：`os.path.join()` 会自动根据操作系统选择正确的路径分隔符，避免在 Windows 上写死 `\` 导致跨平台问题。

### 4.3 pathlib 模块（现代推荐方式）

Python 3.4+ 引入的 `pathlib` 提供了**面向对象**的路径操作，更简洁直观。

```python
from pathlib import Path

# 创建路径对象（用 / 运算符拼接路径）
p = Path('data') / 'subdir' / 'file.txt'

print(p.name)           # file.txt（文件名）
print(p.parent)         # data/subdir（父目录）
print(p.suffix)         # .txt（扩展名）
print(p.exists())       # False（文件是否存在）
print(p.resolve())      # 绝对路径

# 一行读取文件内容
content = p.read_text(encoding='utf-8')

# 一行写入文件
p.write_text('Hello pathlib!', encoding='utf-8')
```

**为什么这样写**：`pathlib` 把路径当作对象处理，支持运算符重载（`/` 拼接），代码更简洁。新项目优先使用 `pathlib`。

### 4.4 os.path vs pathlib 对比

| 操作 | `pathlib`（推荐） | `os.path`（旧式） |
|------|------------------|-------------------|
| 路径拼接 | `p / 'file.txt'` | `os.path.join(p, 'file.txt')` |
| 获取文件名 | `p.name` | `os.path.basename(p)` |
| 获取目录 | `p.parent` | `os.path.dirname(p)` |
| 获取扩展名 | `p.suffix` | `os.path.splitext(p)` |
| 绝对路径 | `p.resolve()` | `os.path.abspath(p)` |
| 是否存在 | `p.exists()` | `os.path.exists(p)` |
| 读取文件 | `p.read_text()` | `open(p).read()` |

---

## 第五章：异常处理——程序的"安全气囊"

### 5.1 什么是异常

异常是程序运行时发生的错误。Python 遇到无法继续执行的情况时，会**抛出异常**。如果不处理，程序就会崩溃并打印错误信息。

**常见异常类型**：

| 异常类型 | 触发场景 |
|----------|----------|
| `FileNotFoundError` | 文件不存在 |
| `PermissionError` | 没有权限访问文件 |
| `IsADirectoryError` | 路径是目录而非文件 |
| `ValueError` | 传入不合适的值（如 `int("abc")`） |
| `IndexError` | 列表索引越界 |
| `KeyError` | 字典中不存在该键 |
| `ZeroDivisionError` | 除以零 |
| `TypeError` | 操作类型不匹配（如 `"1" + 2`） |

### 5.2 try-except 基础结构

```python
try:
    # 可能出错的代码
    with open('data.txt', 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    # 文件不存在时执行
    print("错误：文件不存在，请检查路径")
```

**执行流程**：
1. 先执行 `try` 里的代码
2. 如果没出错，跳过 `except`
3. 如果出错，匹配对应的 `except`，执行其中的代码

### 5.3 捕获多种异常

**分开捕获（推荐，精准处理）**：

```python
try:
    num = int(input("请输入一个数字："))
    result = 100 / num
    with open('data.txt', 'r') as f:
        data = f.read()
except ValueError:
    print("输入的不是有效数字")
except ZeroDivisionError:
    print("不能除以零")
except FileNotFoundError:
    print("文件不存在")
```

**统一捕获多个异常**：

```python
try:
    # 一些操作
    pass
except (ValueError, TypeError) as e:
    print(f"值错误或类型错误：{e}")
```

### 5.4 完整结构：try-except-else-finally

```python
try:
    print("尝试执行...")
    result = 10 / 2
except ZeroDivisionError:
    print("除零错误！")
else:
    # try 中没有异常时执行
    print(f"计算成功，结果是 {result}")
finally:
    # 无论是否异常，最终都会执行（常用于清理资源）
    print("清理工作完成")
```

**输出**：
```
尝试执行...
计算成功，结果是 5.0
清理工作完成
```

**为什么这样写**：
- `else`：把"正常逻辑"和"异常处理"分离，代码更清晰
- `finally`：确保某些操作（如关闭数据库连接、释放锁）一定会执行

### 5.5 获取异常详细信息

```python
try:
    with open('不存在的文件.txt', 'r') as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"异常类型：{type(e).__name__}")
    print(f"异常信息：{e}")
```

### 5.6 主动抛出异常 raise

当函数检测到不符合预期的状态时，可以主动抛出异常，让调用者处理。

```python
def divide(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(f"计算错误：{e}")
```

**为什么这样写**：`raise` 让错误在**最早发现问题的位置**被抛出，而不是带着错误数据继续执行，导致后面更难排查。

### 5.7 自定义异常

通过继承 `Exception` 类，创建具有业务语义的异常。

```python
class InvalidAgeError(Exception):
    """年龄无效时抛出的异常"""
    def __init__(self, age, message="年龄必须在 0~150 之间"):
        self.age = age
        self.message = message
        super().__init__(self.message)

def set_age(age):
    if not (0 <= age <= 150):
        raise InvalidAgeError(age)
    print(f"年龄设置为：{age}")

try:
    set_age(200)
except InvalidAgeError as e:
    print(f"错误：{e.message}，你输入了：{e.age}")
```

**为什么这样写**：自定义异常让错误信息更具可读性，调用者可以通过 `except InvalidAgeError` 精准捕获，而不是捕获宽泛的 `Exception`。

### 5.8 异常链 raise from

在异常处理中抛出另一个异常时，用 `raise from` 保留原始异常信息。

```python
try:
    int('abc')
except ValueError as e:
    raise RuntimeError("数据转换失败") from e
```

---

## 第六章：文件操作 + 异常处理的实战模式

### 6.1 模式一：安全的文件读取函数

```python
def safe_read_file(filepath, encoding='utf-8'):
    """
    安全地读取文本文件，返回 (是否成功, 内容, 错误信息)
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
        return True, content, ""
    except FileNotFoundError:
        return False, "", f"文件 '{filepath}' 不存在"
    except PermissionError:
        return False, "", f"没有权限读取 '{filepath}'"
    except UnicodeDecodeError:
        return False, "", f"文件编码错误，尝试用其他编码读取"
    except Exception as e:
        return False, "", f"未知错误：{type(e).__name__}: {e}"

# 使用
success, content, error = safe_read_file('data.txt')
if success:
    print(content)
else:
    print(f"读取失败：{error}")
```

**为什么这样写**：返回状态元组让调用者完全掌控后续逻辑，而不是被异常强制中断。

### 6.2 模式二：读取 JSON 配置文件

```python
import json

def read_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)   # 直接解析为 Python 字典
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON 格式错误：{e}")
        return {}

config = read_config('settings.json')
print(config.get('learning_rate', 0.001))   # 安全取值，带默认值
```

### 6.3 模式三：逐行处理大文件

```python
def process_large_file(filepath):
    """处理大文件，逐行读取，内存友好"""
    line_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    # 处理每一行的逻辑
                    data = line.strip()
                    if data:
                        print(f"第{line_count}行：{data}")
                except Exception as e:
                    print(f"第{line_count}行处理出错：{e}")
                    continue   # 跳过错误行，继续处理
    except FileNotFoundError:
        print(f"文件 {filepath} 不存在")
    return line_count

process_large_file('big_data.txt')
```

### 6.4 模式四：读取前备份

```python
import shutil
from pathlib import Path
from datetime import datetime

def backup_and_read(filepath):
    """读取文件前先创建备份"""
    src = Path(filepath)
    if not src.exists():
        return False, "文件不存在"

    # 创建备份
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)   # 目录不存在则创建

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f"{src.name}.{timestamp}.bak"
    
    try:
        shutil.copy2(filepath, backup_path)
        print(f"已创建备份：{backup_path}")
        
        content = src.read_text(encoding='utf-8')
        return True, content
    except PermissionError:
        return False, "没有权限"
    except Exception as e:
        return False, str(e)

success, result = backup_and_read('important.txt')
if success:
    print(result)
```

---

## 第七章：常见陷阱与最佳实践

| 坑点 | 正确做法 |
|------|----------|
| 忘记指定 `encoding='utf-8'` | 所有文本文件操作都显式指定编码 |
| 用 `'w'` 模式打开日志文件 | 日志用 `'a'` 追加模式 |
| 大文件用 `read()` 一次性读取 | 用 `for line in f` 逐行迭代 |
| 手动 `open()` 后忘记 `close()` | 始终使用 `with` 语句 |
| 路径里硬编码 `\` 或 `/` | 用 `os.path.join()` 或 `pathlib.Path` |
| 裸 `except:` 捕获所有异常 | 捕获具体异常类型，或至少 `except Exception` |
| 先 `os.path.exists()` 再打开 | 直接用 `try-except`，避免竞态条件 |
| 读取文件后不做编码错误处理 | 捕获 `UnicodeDecodeError`，尝试备用编码 |

---

## 综合实战：学生成绩管理系统（文件版）

```python
import json
from pathlib import Path

DATA_FILE = 'students.json'

def load_data():
    """加载学生数据"""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}   # 文件不存在，返回空字典
    except json.JSONDecodeError:
        print("数据文件损坏，已重置")
        return {}

def save_data(data):
    """保存学生数据"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except PermissionError:
        print("没有权限写入文件")
        return False

def add_student(student_id, name, scores):
    """添加学生"""
    data = load_data()
    if student_id in data:
        print(f"学号 {student_id} 已存在")
        return False
    
    data[student_id] = {
        'name': name,
        'scores': scores,
        'average': sum(scores) / len(scores)
    }
    return save_data(data)

def query_student(student_id):
    """查询学生"""
    data = load_data()
    student = data.get(student_id)
    if student:
        print(f"姓名：{student['name']}")
        print(f"成绩：{student['scores']}")
        print(f"平均分：{student['average']:.2f}")
    else:
        print(f"学号 {student_id} 不存在")

# 使用
add_student('1001', '张三', [85, 92, 78])
add_student('1002', '李四', [90, 88, 95])
query_student('1001')
```

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| `open()` | 完整签名包含 `file, mode, buffering, encoding, errors, newline` 等参数 |
| 文件模式 | `'r'` 读、`'w'` 覆盖写、`'a'` 追加、`'r+'` 读写、`'w+'` 读写清空、`'rb'`/`'wb'` 二进制 |
| `with` | 上下文管理器，确保文件自动关闭，工程安全底线 |
| 读取方式 | `read()` 小文件、`readlines()` 中文件、`for line in f` 大文件 |
| `pathlib` | 现代路径操作，面向对象，优先使用 |
| `try-except` | 捕获具体异常，精准处理 |
| `else/finally` | 正常逻辑放 `else`，清理工作放 `finally` |
| 自定义异常 | 继承 `Exception`，让错误语义化 |

**养成的好习惯**：
1. **所有文件操作都用 `with`**
2. **所有文本操作都写 `encoding='utf-8'`**
3. **捕获具体异常，不用裸 `except:`**
4. **大文件逐行读取，不一次性载入内存**
5. **重要文件操作前先备份**

**下一步预告**：掌握了文件与异常处理后，你将进入 **装饰器与生成器** 的学习——理解 Python 中那些"看似魔法"的语法糖，掌握 `@decorator` 和 `yield` 的底层原理，让你的代码更加 Pythonic。继续加油！