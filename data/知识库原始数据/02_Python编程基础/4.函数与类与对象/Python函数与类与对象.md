# 函数与类与对象：从代码复用到面向对象思维

> **参考文章链接**
> 
> ① 底稿1：《Python 之面向对象详解：从入门到实践》
> 　　网址：https://blog.csdn.net/2501_93253814/article/details/153825790
> 
> ② 底稿2：《〖Python〗函数详细介绍，附代码详解（参数类型、返回值、作用域、高阶函数、闭包、装饰器、生成器、递归等）》
> 　　网址：https://blog.csdn.net/GH_echo/article/details/145823272
> 
> ③ 参考文章1：《第3篇：封装——公有、私有与property》
> 　　网址：https://blog.csdn.net/sundong_3/article/details/160334027
> 
> ④ 参考文章2：《Python 面向对象编程：类与对象、继承、多态完整指南》
> 　　网址：https://blog.csdn.net/2601_96493715/article/details/163083157

---

**适合人群**：已掌握 Python 基础语法和数据结构，希望系统学习函数封装和面向对象编程的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够编写可复用的函数、设计清晰的类结构、理解继承与多态，为后续"文件与异常处理""深度学习框架"打下坚实基础。

**预计学习时长**：约 2.5 小时

---

## 第一部分：函数——代码的"乐高积木"

### 第一章：函数基础定义与调用

#### 1.1 为什么要用函数

假设你需要计算 3 个长方形的面积：

```python
# 不用函数：重复写同样的逻辑
width1, height1 = 3, 4
area1 = width1 * height1

width2, height2 = 5, 6
area2 = width2 * height2

width3, height3 = 7, 8
area3 = width3 * height3
```

这段代码能跑，但**又臭又长**。如果计算逻辑变了（比如要加单位换算），三处都要改。**函数**的作用就是把重复逻辑打包成一个"积木块"，需要时直接调用。

```python
# 定义函数
def calculate_area(width, height):
    """计算长方形面积"""
    return width * height

# 调用函数
area1 = calculate_area(3, 4)
area2 = calculate_area(5, 6)
area3 = calculate_area(7, 8)
```

**为什么这样写**：
- `def` 是"define"的缩写，表示定义函数。
- `calculate_area` 是函数名，用**小写+下划线**命名（行业规范）。
- `width` 和 `height` 是**形参**（形式参数），占位用的。
- `return` 把计算结果"交还"给调用者。
- 函数定义后不会自动执行，必须**调用**才会运行。

#### 1.2 函数定义的基本结构

```python
def 函数名(参数1, 参数2, ...):
    """文档字符串（docstring）：说明函数功能"""
    # 函数体
    return 返回值  # 可选，没有return则默认返回None
```

**示例：无返回值的函数**

```python
def greet(name):
    """向指定的人打招呼"""
    print(f"你好，{name}！")

greet("Alice")   # 输出：你好，Alice！
result = greet("Bob")
print(result)    # None，因为没有return
```

---

### 第二章：参数类型——函数的"入口设计"

函数的灵活程度，很大程度上取决于参数设计。Python 提供了多种参数类型，满足不同的调用需求。

#### 2.1 位置参数

最常见的参数类型，按**位置顺序**传递。

```python
def introduce(name, age, city):
    print(f"我叫{name}，今年{age}岁，来自{city}")

introduce("张三", 20, "北京")
# 输出：我叫张三，今年20岁，来自北京
```

**为什么这样写**：位置参数简单直观，但必须按顺序传，记错顺序就会出错。

#### 2.2 默认参数

给参数设置**默认值**，调用时如果不传，就用默认值。

```python
def introduce(name, age, city="北京"):
    print(f"我叫{name}，今年{age}岁，来自{city}")

introduce("张三", 20)           # 来自北京（使用默认值）
introduce("李四", 22, "上海")   # 来自上海（覆盖默认值）
```

**⚠️ 避坑提醒**：默认参数必须写在位置参数后面！

```python
# ❌ 错误：默认参数不能在前
def wrong(city="北京", name, age):
    pass

# ✅ 正确：位置参数在前，默认参数在后
def correct(name, age, city="北京"):
    pass
```

**⚠️ 另一个大坑**：默认参数如果是可变对象（如列表），会在多次调用间共享！

```python
# ❌ 错误示范
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item(1))   # [1]
print(add_item(2))   # [1, 2]  ← 咦？怎么保留了上次的1！

# ✅ 正确做法：用None做默认值
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

**为什么这样写**：Python 的默认参数在函数定义时就被求值（只执行一次），如果默认参数是可变对象，后续调用会修改同一个对象。

#### 2.3 关键字参数

通过**参数名**传递，可以无视顺序。

```python
def introduce(name, age, city):
    print(f"我叫{name}，今年{age}岁，来自{city}")

# 关键字参数：不用记顺序
introduce(age=20, city="北京", name="张三")
```

**为什么这样写**：参数多时，关键字参数让代码可读性大幅提升，不用回头查函数定义。

#### 2.4 可变参数：*args

当不确定会有多少个位置参数时，用 `*args` 接收。

```python
def total(*numbers):
    """接收任意数量的数字，返回总和"""
    result = 0
    for num in numbers:
        result += num
    return result

# 调用方式极其灵活
print(total(1, 2, 3))      # 6
print(total(10, 20))       # 30
print(total())             # 0
```

**为什么这样写**：`*numbers` 会把所有多余的位置参数**打包成一个元组**。函数内部可以遍历这个元组。

```python
def demo(*args):
    print(args)        # (1, 2, 3)
    print(type(args))  # <class 'tuple'>

demo(1, 2, 3)
```

#### 2.5 可变参数：**kwargs

当不确定会有多少个关键字参数时，用 `**kwargs` 接收。

```python
def print_info(**kwargs):
    """接收任意数量的关键字参数"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="张三", age=20, city="北京")
# 输出：
# name: 张三
# age: 20
# city: 北京
```

**为什么这样写**：`**kwargs` 会把所有多余的关键字参数**打包成一个字典**。

```python
def demo(**kwargs):
    print(kwargs)        # {'name': '张三', 'age': 20}
    print(type(kwargs))  # <class 'dict'>

demo(name="张三", age=20)
```

#### 2.6 参数组合顺序（必须遵守）

定义函数时，参数必须按以下顺序排列：

```python
def func(位置参数, 默认参数, *args, 仅限关键字参数, **kwargs):
    pass

# 示例
def full_demo(a, b=10, *args, c, **kwargs):
    print(f"a={a}, b={b}, args={args}, c={c}, kwargs={kwargs}")

full_demo(1, 2, 3, 4, c=5, d=6, e=7)
# 输出：a=1, b=2, args=(3, 4), c=5, kwargs={'d': 6, 'e': 7}
```

**为什么这样写**：这是 Python 语法规定的顺序，违反会报 `SyntaxError`。

---

### 第三章：返回值——函数的"产出"

#### 3.1 返回单个值

```python
def square(x):
    return x ** 2

result = square(5)
print(result)   # 25
```

#### 3.2 返回多个值

Python 函数可以返回多个值，本质上是返回一个**元组**。

```python
def get_min_max(numbers):
    return min(numbers), max(numbers)

minimum, maximum = get_min_max([3, 1, 4, 1, 5])
print(minimum, maximum)   # 1 5
```

**为什么这样写**：`return a, b` 等价于 `return (a, b)`，调用时用多个变量接收，就是元组解包。

#### 3.3 没有 return

如果函数没有 `return`，默认返回 `None`。

```python
def say_hello():
    print("Hello")

result = say_hello()
print(result)   # None
```

---

### 第四章：作用域——变量的"活动范围"

#### 4.1 局部变量 vs 全局变量

```python
count = 0   # 全局变量

def increment():
    count = 1   # 这是局部变量，和外面的count不是同一个！
    print(f"函数内：{count}")

increment()       # 函数内：1
print(f"函数外：{count}")   # 函数外：0
```

**为什么这样写**：函数内部赋值变量时，Python 默认创建局部变量，不会影响全局变量。

#### 4.2 在函数内修改全局变量

如果确实需要修改全局变量，用 `global` 声明。

```python
count = 0

def increment():
    global count   # 声明：我要用全局的count
    count += 1
    print(f"函数内：{count}")

increment()       # 函数内：1
print(f"函数外：{count}")   # 函数外：1
```

**为什么这样写**：`global` 告诉 Python"不要创建局部变量，去外面找"。但**不建议滥用 global**，它会让代码难以追踪，容易引发 Bug。

#### 4.3 嵌套函数与 nonlocal

```python
def outer():
    x = 10
    
    def inner():
        nonlocal x   # 修改外层（非全局）的x
        x += 1
        print(f"inner: {x}")
    
    inner()
    print(f"outer: {x}")

outer()
# inner: 11
# outer: 11
```

---

### 第五章：匿名函数 lambda

`lambda` 用于创建**小型、一次性**的匿名函数。

```python
# 普通函数
def add(x, y):
    return x + y

# 等价lambda
add = lambda x, y: x + y

print(add(3, 5))   # 8
```

**为什么这样写**：`lambda` 函数没有名字，只能写一行表达式，适合作为参数传给高阶函数。

```python
# 按字符串长度排序
words = ["apple", "fig", "banana", "kiwi"]
words.sort(key=lambda s: len(s))
print(words)   # ['fig', 'kiwi', 'apple', 'banana']
```

---

### 第六章：高阶函数

**高阶函数**是指接收函数作为参数，或返回函数的函数。

#### 6.1 map()——逐个变换

```python
numbers = [1, 2, 3, 4, 5]

# 把每个元素平方
squares = map(lambda x: x ** 2, numbers)
print(list(squares))   # [1, 4, 9, 16, 25]
```

**为什么这样写**：`map(函数, 可迭代对象)` 会把函数应用到每个元素上，返回一个迭代器。

#### 6.2 filter()——筛选

```python
numbers = [1, 2, 3, 4, 5, 6]

# 只保留偶数
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))   # [2, 4, 6]
```

#### 6.3 sorted()——自定义排序

```python
students = [("张三", 85), ("李四", 92), ("王五", 78)]

# 按成绩排序（索引1）
sorted_by_score = sorted(students, key=lambda s: s[1], reverse=True)
print(sorted_by_score)   # [('李四', 92), ('张三', 85), ('王五', 78)]
```

#### 6.4 reduce()——累积计算

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 累乘
product = reduce(lambda x, y: x * y, numbers)
print(product)   # 120（1*2*3*4*5）
```

---

### 第七章：闭包与装饰器（进阶预览）

**闭包**：函数内部定义函数，内部函数引用了外部函数的变量。

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n   # 引用了外部函数的n
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
```

**装饰器**：在不修改原函数代码的情况下，给函数添加额外功能。

```python
def my_decorator(func):
    def wrapper():
        print("函数执行前")
        func()
        print("函数执行后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# 输出：
# 函数执行前
# Hello!
# 函数执行后
```

> 注：装饰器和生成器在知识清单中有单独章节，这里仅做预览性介绍，后续会深入讲解。

---

## 第二部分：面向对象——模仿现实世界的编程思维

### 第八章：为什么需要面向对象

假设要管理学生信息，传统写法（面向过程）：

```python
student1 = ["张三", 17]
student2 = ["李四", 20]

def is_adult(student):
    if student[1] >= 18:
        return f"{student[0]}已成年"
    else:
        return f"{student[0]}未成年"

print(is_adult(student1))
print(is_adult(student2))
```

**问题**：
- 数据和功能分离，学生信息是列表，判断成年的函数在外面
- 扩展性差，新增学生要写新列表
- `student[1]` 这种写法可读性差，不知道1代表什么

面向对象写法：

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def is_adult(self):
        if self.age >= 18:
            return f"{self.name}已成年"
        else:
            return f"{self.name}未成年"

s1 = Student("张三", 17)
s2 = Student("李四", 20)

print(s1.is_adult())   # 张三未成年
print(s2.is_adult())   # 李四已成年
```

**优势**：
- 数据（name, age）和功能（is_adult）绑定在一起
- 新增学生只需 `Student("王五", 19)`
- `s1.is_adult()` 一看就知道是"判断s1是否成年"

---

### 第九章：类与对象

#### 9.1 核心概念

- **类（Class）**：相当于"月饼模子"，是一个模板，定义了对象的属性和方法
- **对象（Object）**：相当于"用模子做的月饼"，是类的实例

在 Python 中，**一切皆对象**。`123` 是 `int` 类的对象，`"hello"` 是 `str` 类的对象。

#### 9.2 定义类

```python
class Student:
    """学生类：存储学生信息并提供相关操作"""
    
    def __init__(self, name, age, grade):
        self.name = name      # 实例属性：姓名
        self.age = age        # 实例属性：年龄
        self.grade = grade    # 实例属性：年级

    def show_info(self):
        """显示学生完整信息"""
        status = "已成年" if self.age >= 18 else "未成年"
        print(f"姓名：{self.name}，年龄：{self.age}，年级：{self.grade}，状态：{status}")

    def is_adult(self):
        """判断是否成年"""
        return self.age >= 18
```

**关键细节**：

| 要素 | 说明 |
|------|------|
| 类名 | 首字母大写，驼峰式（如 `Student`、`UserInfo`） |
| `__init__` | 初始化方法，在对象**创建之后被自动调用**，用来给对象设置初始属性 |
| `self` | 代表"当前对象"，通过 `self.属性名` 绑定属性 |
| 方法 | 这里指**实例方法**：按约定第一个参数是 `self`（名字可改，但约定用 `self`）。注意并非所有方法都有 `self`——类方法（`@classmethod`）第一个参数是 `cls`，静态方法（`@staticmethod`）则既不要 `self` 也不要 `cls` |

**为什么这样写**：`__init__` 常被叫作"构造方法"，但更准确地说它是**初始化方法**——它**不负责创建对象**。真正创建实例（分配内存、生成对象）的是 `__new__`；Python 先调用 `__new__` 造出对象，再把这个对象作为 `self` 传给 `__init__` 做初始化。以 `Student("张三", 17, "高二")` 为例，解释器先执行 `__new__` 得到实例，随后自动调用 `__init__`，把传入的参数变成对象的属性。日常开发几乎只需重写 `__init__`，但要清楚二者分工。

#### 9.3 创建对象（实例化）

```python
# 创建3个学生对象
s1 = Student("张三", 17, "高二")
s2 = Student("李四", 20, "大二")
s3 = Student("王五", 19, "大一")

# 调用对象的方法
s1.show_info()   # 姓名：张三，年龄：17，年级：高二，状态：未成年
print(s2.is_adult())   # True

# 访问对象的属性
print(s1.name)   # 张三
```

**为什么这样写**：`s1 = Student("张三", 17, "高二")` 这行代码做了三件事：
1. 创建一个新的 `Student` 对象
2. 调用 `__init__` 方法，把 "张三"、17、"高二" 赋给 `self.name`、`self.age`、`self.grade`
3. 把创建好的对象赋值给 `s1`

---

### 第十章：封装——公有、私有与 property

#### 10.1 什么是封装

封装有两层含义：
1. **数据与行为捆绑**：把属性（数据）和方法（行为）打包成一个整体
2. **信息隐藏**：对外只暴露必要的接口，隐藏内部实现细节

#### 10.2 公有属性

默认情况下，Python 类的所有属性都是**公有**的，外部可以直接访问。

```python
class Person:
    def __init__(self, name):
        self.name = name   # 公有属性

p = Person("张三")
print(p.name)      # 张三
p.name = "李四"    # 可以直接修改
```

#### 10.3 受保护属性（单下划线）

```python
class Person:
    def __init__(self, name):
        self._name = name   # 受保护属性：暗示"不建议外部直接访问"

p = Person("张三")
print(p._name)     # 可以访问，但Python不会强制阻止
```

**为什么这样写**：单下划线 `_name` 是一种**命名约定**，告诉其他程序员"这个属性是内部使用的，请不要直接碰"。Python 不会强制阻止访问。

#### 10.4 私有属性（双下划线）

```python
class Account:
    def __init__(self, balance):
        self.__balance = balance   # 私有属性

acc = Account(100)
# print(acc.__balance)      # ❌ AttributeError!
print(acc._Account__balance)  # 100（不推荐直接访问）
```

**为什么这样写**：双下划线会触发**名称修饰（Name Mangling）**，`__balance` 被改名为 `_Account__balance`。这主要是为了防止子类属性冲突，并非真正的"私有"。

#### 10.5 @property 装饰器

有时候我们希望属性像变量一样访问，但内部有逻辑控制（如验证）。`@property` 可以把方法变成属性调用。

```python
class Temperature:
    def __init__(self, celsius):
        # 用 self.celsius（属性）赋值，会走下面的 setter 校验，
        # 而不是直接写 self._celsius = celsius 绕过校验
        self.celsius = celsius

    @property
    def celsius(self):
        """获取温度"""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """设置温度（带验证）"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度！")
        self._celsius = value

    @property
    def fahrenheit(self):
        """华氏温度（只读属性）"""
        return self._celsius * 9/5 + 32

# 使用
t = Temperature(25)
print(t.celsius)       # 25（像访问属性一样）
t.celsius = 30         # 调用setter
print(t.fahrenheit)    # 86.0
# t.celsius = -300     # ValueError!（赋值时触发校验）
# Temperature(-300)    # 同样会 ValueError！因为 __init__ 里用的是 self.celsius，
                       # 也会经过 setter 校验
```

**为什么这样写**：
- `@property` 把 `celsius()` 方法变成只读属性 `t.celsius`
- `@celsius.setter` 定义赋值时的逻辑（如验证）
- 没有 setter 的属性就是**只读属性**（如 `fahrenheit`）
- **关键**：`__init__` 中要写 `self.celsius = celsius`（走属性）而**不是** `self._celsius = celsius`（直接写底层字段）。否则构造对象时会**绕过 setter 校验**，像 `Temperature(-300)` 这种非法值就能被创建出来，校验形同虚设

---

### 第十一章：继承——代码复用的艺术

#### 11.1 什么是继承

继承允许子类"继承"父类的属性和方法，避免重复代码。

```python
# 父类（基类）
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "动物发出声音"

# 子类（派生类）
class Dog(Animal):   # Dog 继承自 Animal
    def speak(self):
        return f"{self.name} 汪汪叫！"

# 使用
my_dog = Dog("旺财")
print(my_dog.speak())   # 旺财 汪汪叫！
print(my_dog.name)      # 旺财（继承了name属性）
```

**为什么这样写**：`class Dog(Animal)` 表示"Dog 是一种 Animal"。Dog 自动拥有 Animal 的所有属性和方法，还可以添加自己的或覆盖父类的。

#### 11.2 super() 调用父类方法

```python
class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name)   # 调用父类的__init__
        self.color = color        # 新增自己的属性

    def speak(self):
        parent = super().speak()  # 调用父类的speak
        return f"{parent}，但{self.name}会喵喵叫！"

cat = Cat("小花", "白色")
print(cat.speak())
# 输出：动物发出声音，但小花会喵喵叫！
```

**为什么这样写**：`super()` 返回父类的代理对象，调用 `super().__init__(name)` 就是执行父类的初始化逻辑，避免在子类里重复写。

#### 11.3 多重继承与方法解析顺序（MRO）

Python 支持一个子类继承多个父类。

```python
class A:
    def show(self):
        print("A.show")

class B(A):
    def show(self):
        print("B.show")
        super().show()

class C(A):
    def show(self):
        print("C.show")
        super().show()

class D(B, C):
    def show(self):
        print("D.show")
        super().show()

# 查看MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

d = D()
d.show()
# D.show
# B.show
# C.show
# A.show
```

**为什么这样写**：`super()` 不是直接调用"父类"，而是按照 **MRO（方法解析顺序）** 调用"下一个类"。MRO 遵循 C3 线性化算法，确保每个类只出现一次。

---

### 第十二章：多态——同一接口，不同实现

多态指**同一个方法在不同对象上有不同行为**。Python 是"鸭子类型"：如果它走起来像鸭子、叫起来像鸭子，那它就是鸭子。

```python
class Bird(Animal):
    def speak(self):
        return f"{self.name} 叽叽喳喳"

class Fish(Animal):
    def speak(self):
        return f"{self.name} 吐泡泡"

# 多态函数：接收任何Animal子类
def animal_sound(animal_obj):
    return animal_obj.speak()

# 同一接口，不同实现
animals = [Dog("小黑"), Cat("小花"), Bird("小黄"), Fish("小金")]
for animal in animals:
    print(animal_sound(animal))
```

**为什么这样写**：`animal_sound()` 不关心传入的是 Dog 还是 Fish，只要它有 `speak()` 方法就能工作。新增新动物类型时，无需修改 `animal_sound()`。

---

### 第十三章：魔法函数

魔法函数是以双下划线开头和结尾的特殊方法，让自定义对象支持 Python 内置操作。

| 魔法函数 | 作用 | 触发方式 |
|----------|------|----------|
| `__init__` | 构造器 | 创建对象时自动调用 |
| `__str__` | 友好的字符串表示 | `print(obj)`、`str(obj)` |
| `__repr__` | 官方字符串表示 | `repr(obj)`、交互式环境 |
| `__len__` | 定义长度 | `len(obj)` |
| `__add__` | 定义 + 运算符 | `obj1 + obj2` |
| `__eq__` | 定义 == 运算符 | `obj1 == obj2` |

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2
print(v3)   # Vector(4, 6)
```

---

### 第十四章：继承 vs 组合

| 维度 | 继承 (is-a) | 组合 (has-a) |
|------|-------------|--------------|
| 关系 | "是一个"（Dog 是 Animal） | "有一个"（Car 有 Engine） |
| 耦合度 | 强耦合 | 松耦合 |
| 灵活性 | 较低 | 较高 |
| 适用场景 | 明确的层次关系 | 组件化、可插拔设计 |

**优先使用组合**，除非确实是"是一个"关系。

---

## 第三部分：综合实战

### 实战一：图形计算系统

```python
import math

class Shape:
    """图形基类"""
    def area(self):
        raise NotImplementedError("子类必须实现 area 方法")

    def perimeter(self):
        raise NotImplementedError("子类必须实现 perimeter 方法")

    def __str__(self):
        return f"{self.__class__.__name__}"

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

    def __str__(self):
        return f"圆形(半径={self.radius})"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

    def __str__(self):
        return f"矩形(宽={self.width}, 高={self.height})"

# 多态使用
shapes = [Circle(5), Rectangle(4, 6), Circle(3)]
for shape in shapes:
    print(f"{shape}: 面积={shape.area():.2f}, 周长={shape.perimeter():.2f}")
```

### 实战二：银行账户封装

```python
class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner
        self.__balance = initial_balance   # 私有属性
        self.__transaction_log = []        # 私有日志

    @property
    def balance(self):
        """只读属性：余额"""
        return self.__balance

    def deposit(self, amount):
        """存款"""
        if amount > 0:
            self.__balance += amount
            self.__log(f"存款 +{amount}")
        else:
            raise ValueError("存款金额必须为正")

    def withdraw(self, amount):
        """取款"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__log(f"取款 -{amount}")
        else:
            raise ValueError("余额不足或金额无效")

    def __log(self, msg):   # 私有方法
        self.__transaction_log.append(msg)

    def get_history(self):
        """获取交易记录"""
        return self.__transaction_log.copy()

# 使用
acc = BankAccount("Alice", 100)
acc.deposit(50)
print(acc.balance)           # 150
print(acc.get_history())     # ['存款 +50']

# 注意：下面这行【不会报错】，也【改不了真正的余额】！
acc.__balance = 999          # 只是在 acc 上新建了一个名为 "__balance" 的普通属性
print(acc.__balance)         # 999  ← 这是你新建的属性
print(acc.balance)           # 150  ← 真正的余额没变
# 真正的余额被名称修饰藏在 _BankAccount__balance 里，和上面的 __balance 是两回事
print(acc._BankAccount__balance)   # 150
```

> **为什么不会报错**：在类**外部**写 `acc.__balance = 999` 时，Python **不会**做名称修饰（名称修饰只发生在类定义体内部）。所以它并没有去动那个被改名为 `_BankAccount__balance` 的私有属性，而是**在实例上新建了一个字面就叫 `__balance` 的新属性**。结果就是：不报错、真正的余额也没被改。这也说明双下划线私有属性并非"锁死不可改"，而是靠改名来**降低误碰概率**。

---

## 本章小结与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 默认参数用可变对象（如 `[]`） | 用 `None` 做默认值，函数内部判断 |
| 参数顺序写错（默认参数在位置参数前） | 位置参数 → 默认参数 → `*args` → 关键字参数 → `**kwargs` |
| 函数内直接修改全局变量 | 用 `global` 声明，或传参返回新值 |
| 混淆 `append()` 和 `extend()` | `append` 加整体，`extend` 拆开加 |
| 忘记 `self` 参数 | 类中所有方法的第一个参数必须是 `self` |
| 双下划线属性直接访问报错 | 用 `_ClassName__attr`（不推荐），或用 `@property` |
| 滥用 `global` | 优先用参数传递和返回值 |
| 继承时忘记 `super().__init__()` | 子类需要父类初始化逻辑时务必调用 |
| 魔法函数命名错误 | 必须是双下划线开头和结尾，如 `__str__` |

**养成的好习惯**：
1. 函数只做一件事，命名要表达清楚功能
2. 优先使用关键字参数提高可读性
3. 类的属性通过 `@property` 控制访问，不要直接暴露
4. 能用组合就不用继承
5. 写文档字符串（docstring）说明类/函数的功能

**下一步预告**：掌握了函数与面向对象后，你将进入 **文件与异常处理** 的学习——学会读写文件、处理程序运行中的错误，让你的程序更加健壮。继续加油！