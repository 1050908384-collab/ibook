# 装饰器与生成器：Python 高阶特性的两大支柱

> **参考文章链接**
> 
> ① 底稿：《Python三大高阶特性详解：装饰器、生成器与上下文管理器》
> 　　网址：https://blog.csdn.net/2301_80190025/article/details/160443766
> 
> ② 参考文章1：《装饰器导致元数据丢失？教你3步用wraps完美修复》
> 　　网址：https://blog.csdn.net/PoliVein/article/details/155265733
> 
> ③ 参考文章2：《Python生成器yield from语义详解》
> 　　网址：https://blog.csdn.net/2501_94114815/article/details/162020610
> 
> ④ 参考文章3：《Python 装饰器从入门到精通》
> 　　网址：https://blog.csdn.net/try_username/article/details/161545167
> 
> ⑤ 参考文章4：《Python 装饰器原理详解》
> 　　网址：https://blog.csdn.net/hxxjxw/article/details/121566882

---

**适合人群**：已掌握 Python 函数、类与对象、文件操作，希望深入理解 Python 高阶特性的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够编写装饰器扩展功能、使用生成器处理海量数据、理解 `yield from` 的底层机制，为后续"PyTorch 进阶训练技巧""分布式训练"打下坚实基础。

**预计学习时长**：约 2 小时

---

## 第一部分：装饰器——不修改代码就能增强功能

### 第一章：装饰器基础——给函数"套壳"

#### 1.1 什么是装饰器

**装饰器（Decorator）** 是一种特殊的函数，它的作用是**在不修改原函数代码的前提下，给函数增加新功能**。就像给手机套一个保护壳——手机本身没变，但多了防摔功能。

**为什么需要装饰器**：假设你有 10 个函数，每个函数执行前后都要打印日志。不用装饰器，你得在每个函数里重复写 `print`。用了装饰器，只需写一次，然后 `@` 一下就能复用。

#### 1.2 最简单的装饰器

```python
# 定义装饰器
def my_decorator(func):
    def wrapper():
        print("函数执行之前")
        func()          # 调用原函数
        print("函数执行之后")
    return wrapper

# 使用装饰器（@语法糖）
@my_decorator
def say_hello():
    print("你好")

say_hello()
```

**输出**：
```
函数执行之前
你好
函数执行之后
```

**为什么这样写**：
- `my_decorator` 接收一个函数 `func` 作为参数（这就是"高阶函数"）
- `wrapper` 是内部函数，在原函数前后添加了新逻辑
- `@my_decorator` 是语法糖，**等价于** `say_hello = my_decorator(say_hello)`
- 调用 `say_hello()` 时，实际执行的是 `wrapper()`

#### 1.3 装饰带参数的函数

上面的装饰器只能包裹无参函数。要包裹任意参数的函数，`wrapper` 需要用 `*args, **kwargs`：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"正在调用函数: {func.__name__}")
        result = func(*args, **kwargs)   # 把参数透传给原函数
        print(f"函数执行完毕")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

print(add(1, 2))   # 输出：3
```

**为什么这样写**：
- `*args` 接收所有位置参数
- `**kwargs` 接收所有关键字参数
- 这样装饰器就能包裹任何函数，不管它有几个参数

---

### 第二章：带参数的装饰器——三层嵌套

有时候装饰器本身也需要参数，比如控制是否打印日志、设置重试次数等。

**核心思路**：装饰器只能接收一个函数参数。如果装饰器自己也要参数，就需要**再包一层函数**。

```python
def log_decorator(enabled=True):
    # 第一层：接收装饰器自己的参数
    def decorator(func):
        # 第二层：接收被装饰的函数
        def wrapper(*args, **kwargs):
            # 第三层：实际的包裹逻辑
            if enabled:
                print(f"[LOG] 调用 {func.__name__}，参数: {args}, {kwargs}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 使用：启用日志
@log_decorator(enabled=True)
def greet(name):
    print(f"你好, {name}")

# 使用：禁用日志
@log_decorator(enabled=False)
def bye(name):
    print(f"再见, {name}")

greet("张三")   # [LOG] 调用 greet...
bye("李四")     # 没有日志输出
```

**为什么这样写**：
- `@log_decorator(enabled=True)` 先执行 `log_decorator(True)`，返回 `decorator` 函数
- 然后 `@decorator` 装饰 `greet`，等价于 `greet = decorator(greet)`
- 最终 `greet` 指向 `wrapper`

**执行顺序记忆口诀**：`@` 后面的表达式先执行，返回值作为真正的装饰器。

---

### 第三章：多装饰器叠加——由里到外包裹

多个装饰器可以同时作用于一个函数，执行顺序是**由内到外**（先执行靠近函数的装饰器）。

```python
def dec_1(func):
    def inner(*args, **kwargs):
        print("装饰器 1 开始")
        result = func(*args, **kwargs)
        print("装饰器 1 结束")
        return result
    return inner

def dec_2(func):
    def inner(*args, **kwargs):
        print("装饰器 2 开始")
        result = func(*args, **kwargs)
        print("装饰器 2 结束")
        return result
    return inner

def dec_3(func):
    def inner(*args, **kwargs):
        print("装饰器 3 开始")
        result = func(*args, **kwargs)
        print("装饰器 3 结束")
        return result
    return inner

@dec_1
@dec_2
@dec_3
def target():
    print("=== 原函数执行 ===")

target()
```

**输出**：
```
装饰器 1 开始
装饰器 2 开始
装饰器 3 开始
=== 原函数执行 ===
装饰器 3 结束
装饰器 2 结束
装饰器 1 结束
```

**为什么这样写**：`@dec_1 @dec_2 @dec_3` 等价于 `target = dec_1(dec_2(dec_3(target)))`。就像俄罗斯套娃，最里面的先执行，然后一层层向外展开。

---

### 第四章：类装饰器——用类来装饰函数

除了函数，**类也可以作为装饰器**。关键是实现 `__init__` 和 `__call__` 两个方法。

```python
class Timer:
    def __init__(self, func):
        # 接收被装饰的函数，保存为实例属性
        self.__func = func

    def __call__(self, *args, **kwargs):
        # 让实例可以像函数一样被调用
        import time
        start = time.time()
        result = self.__func(*args, **kwargs)
        print(f"{self.__func.__name__} 执行耗时: {time.time() - start:.4f}秒")
        return result

@Timer
def slow_function():
    import time
    time.sleep(0.5)
    return "完成"

slow_function()   # 自动计时
```

**为什么这样写**：
- `@Timer` 等价于 `slow_function = Timer(slow_function)`，创建了一个 `Timer` 实例
- 调用 `slow_function()` 时，Python 发现它是对象而非函数，自动调用 `__call__`
- `__call__` 里执行计时逻辑，再调用 `self.__func`

**⚠️ 避坑提醒**：类装饰器必须实现 `__call__`，否则实例无法被调用，会报 `TypeError: 'Timer' object is not callable`。

---

### 第五章：修复元数据——functools.wraps

#### 5.1 元数据丢失问题

装饰器有一个隐蔽的 Bug：被装饰后，原函数的"身份信息"会丢失。

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """向用户打招呼"""
    return f"Hello, {name}"

print(greet.__name__)   # wrapper  ← 不是 greet！
print(greet.__doc__)    # None     ← 文档字符串丢了！
```

**为什么这样写**：`greet` 被替换成了 `wrapper`，所以 `greet.__name__` 显示的是 `wrapper`。这在调试、文档生成、框架反射时会造成严重困扰。

#### 5.2 用 @wraps 修复

`functools.wraps` 是一个专门修复这个问题的装饰器，它会把原函数的元数据复制给 `wrapper`。

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)   # ← 加上这一行
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """向用户打招呼"""
    return f"Hello, {name}"

print(greet.__name__)   # greet  ← 正确！
print(greet.__doc__)    # 向用户打招呼  ← 正确！
```

**为什么这样写**：`@wraps(func)` 内部调用了 `update_wrapper(wrapper, func)`，把 `func` 的 `__name__`、`__doc__`、`__module__`、`__qualname__`、`__annotations__` 等属性复制给 `wrapper`。

#### 5.3 对比总结

| 装饰方式 | `__name__` | `__doc__` | 是否影响调试 |
|----------|-----------|-----------|-------------|
| 未使用 wraps | ❌ 变成 wrapper | ❌ 丢失 | ✅ 严重影响 |
| 使用 `@wraps(func)` | ✅ 保留原值 | ✅ 保留原值 | ❌ 无影响 |

**养成的好习惯**：**写任何自定义装饰器时，wrapper 上都必须加 `@wraps(func)`**。这是工程规范，不是可选的。

---

## 第二部分：迭代器与生成器——惰性计算的利器

### 第六章：迭代器——逐个"拧水龙头"

#### 6.1 什么是迭代器

迭代器是一种对象，它可以**逐个访问集合中的元素**，而不需要一次性把所有元素加载到内存。你可以把迭代器理解为一个水龙头——水一滴一滴流出来，而不是一桶水直接倒出来。

#### 6.2 迭代器的协议

一个对象要成为迭代器，必须实现两个方法：

| 方法 | 作用 |
|------|------|
| `__iter__()` | 返回迭代器对象本身 |
| `__next__()` | 返回下一个元素，没有时抛出 `StopIteration` |

```python
class CountUp:
    def __init__(self, max_num):
        self.max_num = max_num
        self.current = 0

    def __iter__(self):
        return self   # 返回迭代器对象本身

    def __next__(self):
        if self.current < self.max_num:
            num = self.current
            self.current += 1
            return num
        else:
            raise StopIteration   # 没有元素了

# 使用
for num in CountUp(5):
    print(num)   # 0 1 2 3 4
```

**为什么这样写**：`for` 循环底层会先调用 `__iter__()` 获取迭代器，然后不断调用 `__next__()` 获取元素，直到遇到 `StopIteration` 停止。

---

### 第七章：生成器——用函数写迭代器

#### 7.1 什么是生成器

生成器是一种**特殊的迭代器**，它用**函数**来定义，而不是类。生成器函数使用 `yield` 关键字，执行到 `yield` 时函数会**暂停**，返回一个值；下次调用时从暂停处**继续执行**。

```python
def my_generator():
    print("开始执行")
    yield 1       # 暂停，返回 1
    print("继续执行")
    yield 2       # 暂停，返回 2
    print("再次继续")
    yield 3       # 暂停，返回 3
    print("执行结束")

# 创建生成器对象（注意：此时函数体不会执行！）
gen = my_generator()

# 手动获取下一个值
print(next(gen))   # 开始执行 → 1
print(next(gen))   # 继续执行 → 2
print(next(gen))   # 再次继续 → 3
# next(gen) 再次调用会抛出 StopIteration
```

**为什么这样写**：
- 调用 `my_generator()` 不会执行函数体，而是返回一个**生成器对象**
- `next(gen)` 触发函数执行，直到遇到 `yield`
- 每次 `yield` 后函数暂停，**状态被保存**（局部变量、执行位置）
- 下次 `next()` 从暂停处继续执行

#### 7.2 用 for 循环遍历生成器

```python
def count_up_to(max_num):
    count = 1
    while count <= max_num:
        yield count
        count += 1

for num in count_up_to(5):
    print(num)   # 1 2 3 4 5
```

#### 7.3 生成器 vs 列表——内存对比

```python
# 列表推导式：一次性生成所有元素，占用大量内存
my_list = [x * 2 for x in range(1000000)]

# 生成器表达式：按需生成，几乎不占内存
my_gen = (x * 2 for x in range(1000000))

print(next(my_gen))   # 0
print(next(my_gen))   # 2
```

**为什么这样写**：生成器表达式 `(x * 2 for x in range(1000000))` 和列表推导式的语法几乎一样，只是把 `[]` 换成了 `()`。但它不会立即计算所有值，而是每次 `next()` 时才计算一个，处理海量数据时内存效率极高。

---

### 第八章：yield from——委托子生成器

`yield from` 是 Python 3.3+ 引入的语法，用于**将子生成器的控制权委托给调用者**。它建立了一条双向通信通道，让调用者与子生成器直接通信。

#### 8.1 基本用法

```python
def sub_gen():
    yield 1
    yield 2
    yield 3

def delegator():
    yield from sub_gen()   # 把 sub_gen 的产出直接交给调用者

for value in delegator():
    print(value)   # 1 2 3
```

**为什么这样写**：`yield from sub_gen()` 等价于手动写：

```python
def delegator_manual():
    gen = sub_gen()
    try:
        while True:
            value = gen.send(None)
            yield value
    except StopIteration:
        pass
```

`yield from` 自动处理了子生成器的启动、迭代和 `StopIteration`，代码更简洁。

#### 8.2 获取子生成器的返回值

```python
def accumulate(max_num):
    total = 0
    for i in range(max_num):
        total += i
        yield i
    return total   # 生成器函数的 return 值！

def collect():
    result = yield from accumulate(5)
    print(f"子生成器返回的总和: {result}")

list(collect())   # 输出: 子生成器返回的总和: 10
```

**为什么这样写**：生成器函数中的 `return value` 会被包装成 `StopIteration(value)`。`yield from` 捕获这个异常后，提取 `value` 作为表达式的结果赋值给 `result`。

#### 8.3 异常通过 yield from 传播

```python
def child():
    try:
        yield 1
        yield 2
    except ZeroDivisionError:
        yield "捕获到除零错误"

def parent():
    yield from child()
    yield "父生成器继续"

p = parent()
print(next(p))              # 1
print(p.throw(ZeroDivisionError))   # 捕获到除零错误
print(next(p))              # 父生成器继续
```

**为什么这样写**：`throw()` 抛出的异常会通过 `yield from` 直接传播到子生成器。子生成器捕获异常后继续执行，`yield from` 仍然保持活动状态。

---

### 第九章：生成器的高级控制——send、close、throw

生成器对象除了 `next()`，还有三个重要方法，用于**双向通信**和**生命周期控制**。

#### 9.1 send()——向生成器传值

`send(value)` 有两个作用：
1. 像 `next()` 一样恢复生成器执行
2. 把 `value` 作为 **yield 表达式的返回值** 传给生成器内部

```python
def echo():
    received = yield "ready"           # yield "ready" 产出 "ready"
    yield f"收到: {received}"          # send 的值赋给 received

g = echo()
print(g.send(None))      # 必须用 None 启动！输出: ready
print(g.send("Hello"))   # 输出: 收到: Hello
```

**⚠️ 避坑提醒**：第一次调用生成器时，必须用 `g.send(None)` 或 `next(g)` 启动。**不能用带参数的 `send(value)` 启动**，因为此时生成器还没遇到 yield，没有地方接收值。

#### 9.2 close()——终止生成器

`close()` 在生成器暂停处抛出 `GeneratorExit` 异常，终止生成器执行。

```python
def foo():
    try:
        yield 1
    except GeneratorExit:
        print("捕获到 GeneratorExit，执行清理")
        raise   # 必须重新抛出，不能吞掉！

g = foo()
print(next(g))   # 1
g.close()        # 输出: 捕获到 GeneratorExit，执行清理
```

**⚠️ 避坑提醒**：
- 捕获 `GeneratorExit` 后**必须重新抛出**，否则 Python 会报 `RuntimeError`
- `close()` 后不能再调用 `next()` 或 `send()`，否则会报 `StopIteration` 或 `RuntimeError`

#### 9.3 throw()——向生成器抛异常

`throw(exception)` 在生成器暂停处抛出指定异常，生成器可以用 `try-except` 捕获。

```python
def foo():
    try:
        yield 1
    except ValueError:
        yield "捕获到 ValueError"

g = foo()
print(next(g))           # 1
print(g.throw(ValueError))  # 捕获到 ValueError
```

**为什么这样写**：`throw()` 常用于外部控制生成器的异常处理逻辑，比如在协程中通知生成器发生了某种错误。

---

## 第三部分：上下文管理器——资源的"自动管家"

### 第十章：上下文管理器——with 语句的底层

上下文管理器用于**管理资源的生命周期**，确保资源在使用前正确打开，使用后正确关闭，无论是否发生异常。

#### 10.1 自定义上下文管理器

通过实现 `__enter__` 和 `__exit__` 两个方法：

```python
class ManagedFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        # 进入 with 块时执行，返回的对象赋值给 as 后面的变量
        print(f"打开文件: {self.filename}")
        self.file = open(self.filename, self.mode, encoding='utf-8')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出 with 块时执行，无论是否异常都会调用
        print(f"关闭文件: {self.filename}")
        self.file.close()
        return False   # False 表示不压制异常，继续向外传播

# 使用
with ManagedFile('test.txt', 'w') as f:
    f.write('Hello, 上下文管理器!')
# 离开 with 块，自动调用 __exit__
```

**为什么这样写**：
- `__enter__` 返回资源对象（如文件对象）
- `__exit__` 接收三个参数：异常类型、异常值、异常追踪信息。如果返回 `True`，表示异常已被处理，不会向外传播；返回 `False` 或 `None`，异常继续传播
- 无论 `with` 块内发生什么（正常结束、return、break、抛出异常），`__exit__` 一定会执行

#### 10.2 用 contextlib 简化

对于简单的场景，可以用 `@contextmanager` 装饰器，无需写类：

```python
from contextlib import contextmanager

@contextmanager
def managed_file(filename, mode):
    f = open(filename, mode, encoding='utf-8')
    try:
        yield f   # yield 之前的代码相当于 __enter__
    finally:
        f.close()  # yield 之后的代码相当于 __exit__

with managed_file('test.txt', 'w') as f:
    f.write('Hello from contextmanager!')
```

**为什么这样写**：`@contextmanager` 把生成器函数转换成上下文管理器。`yield` 之前的代码在进入时执行，`yield` 之后的代码在退出时执行（即使发生异常）。

---

## 综合实战：日志装饰器 + 生成器处理大文件

### 实战一：带参数的重试装饰器

```python
from functools import wraps
import time

def retry(max_attempts=3, delay=1):
    """带参数的装饰器：失败时自动重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"第 {attempt} 次尝试失败: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("模拟随机失败")
    return "成功！"

print(unstable_function())
```

### 实战二：生成器逐行处理大日志文件

```python
def log_parser(filepath):
    """生成器：逐行解析日志，内存友好"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                # 假设日志格式: [时间] [级别] 消息
                parts = line.split(']', 2)
                timestamp = parts[0][1:]
                level = parts[1][1:].strip()
                message = parts[2].strip()
                yield {
                    'line': line_num,
                    'timestamp': timestamp,
                    'level': level,
                    'message': message
                }
            except IndexError:
                yield {'line': line_num, 'error': '格式错误', 'raw': line}

# 使用：只提取 ERROR 级别的日志
error_logs = (log for log in log_parser('app.log') if log.get('level') == 'ERROR')
for log in error_logs:
    print(f"第{log['line']}行: {log['message']}")
```

### 实战三：嵌套生成器处理多级目录

```python
from pathlib import Path

def find_files(root, pattern='*.py'):
    """递归查找文件，使用 yield from 委托"""
    for item in Path(root).iterdir():
        if item.is_dir():
            yield from find_files(item, pattern)   # 委托子生成器
        elif item.match(pattern):
            yield item

# 使用
for py_file in find_files('.'):
    print(py_file)
```

---

## 本章小结与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 装饰器没加 `@wraps(func)` | wrapper 上必须加 `@wraps(func)`，保留元数据 |
| 带参数装饰器只写两层 | 需要三层嵌套：外层收参数 → 中层收函数 → 内层执行逻辑 |
| 多装饰器顺序搞反 | 执行顺序由内到外，靠近函数的最先执行 |
| 类装饰器没写 `__call__` | 必须实现 `__call__`，否则实例不可调用 |
| 用 `send(value)` 启动生成器 | 第一次必须用 `send(None)` 或 `next()` |
| 捕获 `GeneratorExit` 不重新抛出 | 必须 `raise`，否则报 `RuntimeError` |
| `close()` 后再调用 `next()` | 生成器已关闭，会报 `StopIteration` |
| 列表推导式处理大数据 | 改用生成器表达式 `(x for x in ...)` 节省内存 |

| 知识点 | 核心要点 |
|--------|----------|
| **装饰器** | 高阶函数，`@decorator` 等价于 `func = decorator(func)` |
| **带参数装饰器** | 三层嵌套：外层参数 → 中层函数 → 内层逻辑 |
| **多装饰器** | 由内到外包裹，执行时先内后外 |
| **类装饰器** | 实现 `__init__` 和 `__call__` |
| **@wraps** | 保留原函数 `__name__`、`__doc__` 等元数据 |
| **迭代器** | 实现 `__iter__` 和 `__next__` |
| **生成器** | 函数中使用 `yield`，自动保存状态 |
| **生成器表达式** | `(x for x in ...)`，惰性计算，省内存 |
| **yield from** | 委托子生成器，自动传播 send/throw/close |
| **send()** | 向生成器传值，第一次必须用 `None` |
| **close()** | 终止生成器，触发 `GeneratorExit` |
| **throw()** | 向生成器抛异常 |
| **上下文管理器** | `__enter__` + `__exit__`，或用 `@contextmanager` |

**养成的好习惯**：
1. 写装饰器必加 `@wraps(func)`
2. 处理大数据必用生成器/生成器表达式
3. 资源管理必用 `with` 语句
4. 复杂嵌套用 `yield from` 简化代码

**下一步预告**：掌握了装饰器与生成器后，你将进入 **PyTorch 基础**的学习——从张量操作到自动求导，从 `torch.nn` 到训练循环，正式踏入深度学习的大门。继续加油！