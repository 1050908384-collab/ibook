# 学习包

## 块 1

## 计算图：Autograd 的“地图”

### 为什么你要先懂这张“地图”？

你的目标是 **QLoRA 微调**——本质上是让一个已经训练好的大模型在你的数据上“再学一点点”。而微调的核心动作，就是**算梯度、更新参数**。Autograd 就是 PyTorch 帮你算梯度的引擎，而计算图就是 Autograd 手里的那张“地图”。不懂这张图，你后面看 LoRA 的梯度流动、看 `detach()` 冻结参数，都会像看天书。

所以这一节，我们只干一件事：**把这张“地图”彻底看明白**。

---

### 2.1 什么是计算图？

一句话：**计算图是 PyTorch 记录张量运算过程的一张有向无环图（DAG）**。

拆开来看，就两个要素：

- **节点（Node）**：张量（Tensor），也就是你手里的数据
- **边（Edge）**：运算操作，比如加减乘除、矩阵乘法

整个流程分两步走：

1. **前向传播（Forward）**：从输入开始，一步步算出输出。PyTorch 在这个过程中**顺手把每一步运算都记了下来**，形成这张图。
2. **反向传播（Backward）**：当你调用 `.backward()` 时，PyTorch 沿着这张图**从后往前**走一遍，用链式法则把每个参数的梯度算出来。

---

### 一个具体的例子

假设你要算这个式子：

\[
y = x_1 \cdot w + x_2
\]

PyTorch 会把运算过程记录成下面这张图：

```
x1 ────┐
        × ──── + ──── y
w  ────┘     ↑
x2 ──────────┘
```

你看，`x1` 和 `w` 先做乘法（`×`），得到一个中间结果；然后这个中间结果再和 `x2` 做加法（`+`），得到最终的 `y`。

PyTorch 把这张图**完整地存了下来**。当你调用 `y.backward()` 时，它就沿着这张图从 `y` 出发，**倒着走**：先算加法对 `x2` 和中间结果的梯度，再算乘法对 `x1` 和 `w` 的梯度。

**这就是 Autograd 的全部秘密：你只管定义前向计算，它自动帮你记图、算梯度。**

---

### 动手验证一下（30秒）

在 `pytorch` conda 环境里跑这段代码：

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor

## 块 2

## 动态计算图：你的“活地图”

PyTorch 用的是**动态计算图**——图在代码**运行时**逐行构建，每次前向传播都可以不一样。这意味着你可以像写普通 Python 代码一样，在模型里用 `if`、`for` 控制流，PyTorch 会根据实际执行路径自动记录操作序列。

```python
import torch

def dynamic_net(x):
    # 动态控制流：根据输入决定走哪条路
    if x.sum() > 0:
        return x * 2
    else:
        return x * 3

x = torch.tensor([1.0, -1.0], requires_grad=True)
y = dynamic_net(x).sum()
y.backward()
print(x.grad)   # 根据实际路径正确计算梯度
```

**为什么这对你的 QLoRA 微调重要？** 微调时模型结构常因输入变化（如不同长度序列）而不同，动态图让你无需预定义完整

## 块 3

### 3.1 requires_grad：标记"谁需要求导"

在 PyTorch 中，**只有**设置了 `requires_grad=True` 的张量，才会被加入计算图、参与梯度计算。这是自动求导的"开关"——你决定谁需要被追踪。

**动手试试**：直接运行下面代码，观察输出。

```python
import torch

# 叶子节点：用户直接创建的张量，通常作为模型参数或输入
x = torch.tensor(2.0, requires_grad=True)   # 需要求导
w = torch.tensor(3.0, requires_grad=True)   # 需要求导
b = torch.tensor(1.0, requires_grad=False)  # 不需要求导（默认）

print(x.is_leaf)   # True  ← 叶子节点
print(w.is_leaf)   # True
```

**关键概念：叶子节点（Leaf Node）**

- 由用户

## 块 4

你正在为 QLoRA 微调打基础，核心是搞懂 PyTorch 怎么“自动”算梯度。这一块我们用最直接的方式，边看代码边理解。

**前向传播：计算图悄悄搭建**

执行下面这行代码时，PyTorch 不只是算了个结果，它还在后台把运算步骤记了下来：

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
b = torch.tensor(1.0)

y = w * x + b
print(y)           # tensor(7., grad_fn=<AddBackward0>)
print(y.grad_fn)   # <AddBackward0>  ← 记录了"y是通过加法得到的"
```

注意 `y` 的 `grad_fn` 属性，它就像计算图上的一个“节点标签”，告诉你 `y` 是由加法操作产生的。这个标签是反向传播的线索，它知道如何对这个运算求导。

**反向传播

## 块 5

你正在为 QLoRA 微调做准备，理解梯度累加是避免训练崩溃的关键。PyTorch 的 `.grad` 属性是**累加**的，不是覆盖。

```python
y = w * x + b
y.backward()
print(w.grad)  # 2.0

# 第二次反向传播（没清零）
y = w * x + b
y.backward()
print(w.grad)  # 4.0 ← 累加了！
```

标准训练中，每次迭代开始必须清零：

```python
w.grad.zero_()
x.grad.zero_()
# 或：optimizer.zero_grad()
```

**动手验证**：跑上面代码，观察两次 `w.grad` 输出，确认累加行为。

## 块 6

## 4.1 torch.no_grad()：整段代码不建图

你要做 QLoRA 微调，但微调之前必须先会“推理”。推理时模型只做预测、不更新参数，这时**不需要计算图**。`torch.no_grad()` 就是干这个的——它是一个上下文管理器，进入后，里面的所有操作都不会被记录到计算图中。这是**推理阶段的标准做法**，能省显存、省计算。

```python
with torch.no_grad():
    pred = model(x)      # 不构建计算图，速度更快，省内存
    accuracy = (pred == label).float().mean()
```

注意：`with` 块结束后，梯度追踪自动恢复。你在 QLoRA 训练循环里，前向传播和评估指标计算都可以包在 `no_grad()` 里，只有反向传播那一步需要计算图。

## 4.2 detach()：单个张量“摘”出计算图

`torch.no_grad()` 是“整段不建图”，而 `.detach()` 是“只摘一个张量”。它返回一个新张量，**与原张量共享数据**，但不再有 `grad_fn`，`requires_grad=False`。它只影响从它开始的新分支，不会破坏原计算图。

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y =

## 块 7

## 4.3 no_grad vs detach vs eval：一张表分清三兄弟

你要做 QLoRA 微调，推理和验证时这三个东西最容易混。直接看表：

| 方法 | 作用范围 | 是否构建计算图 | 典型场景 |
|---|---|---|---|
| `torch.no_grad()` | 整段代码块 | 不构建 | 模型推理、验证 |
| `.detach()` | 单个张量 | 该张量后续不参与梯度计算 | 冻结子网络、转 NumPy |
| `model.eval()` | 模型行为 | 与梯度无关 | 关闭 Dropout/BatchNorm 训练模式 |

**核心区别一句话**：`no_grad()` 管的是“

## 块 8

## 5.1 核心区别：叶子节点 vs 非叶子节点

在自动求导中，**叶子节点**是你亲手创建的张量，**非叶子节点**是运算产生的中间结果。判断方法很简单：看 `is_leaf` 属性——叶子节点为 `True`，非叶子节点为 `False`。

**关键差异在梯度存储**：叶子节点的 `.grad` 会保存梯度，而非叶子节点默认不保存（为了省内存）。比如 `y = x * w` 中的 `y` 就是非叶子节点，你直接访问 `y.grad` 会得到 `None`。

**动手验证**：如果你确实想看中间结果的梯度，用 `.retain_grad()` 显式保留：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.retain_grad()    # 显式保留 y 的梯度
z = y * 3
z.backward()

print(y.grad)      # 3.0  ← 没有 retain_grad() 时这里是 None
print(x.grad)      # 12.0
```

## 六、标准训练循环：Autograd 实战

把知识串起来，一个典型的 PyTorch 训练循环长这样：

```python
import torch
import torch.nn as nn

# 准备数据
x = torch.randn(64, 10)
y = torch.randn(64, 1)

model = nn.Linear(10, 1)           # 简单线性模型
optimizer = torch.optim.SGD(model

## 块 9

**手动验证链式法则（面向QLoRA微调）**

你正为QLoRA微调打基础，这里用手动计算验证PyTorch反向传播的链式法则——这是理解梯度如何流经模型参数的关键。

**代码与验证**

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w1 = torch.tensor(3.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)

h = x * w1          # h = 6
y = h ** 2 + w2     # y = 40

y.backward()
print(x.grad, w1.grad, w2.grad)
```

**手动推导**（对照输出检查）：

- `

## 块 10

**本块学习检查**：运行下方代码，观察三次 `backward()` 后 `x.grad` 的变化，并记录每次打印的值。

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第{i+1}次 backward 后, x.grad = {x.grad}")
```

**核心要点**：每次 `backward()` 都会在原有梯度上**累加**新值，而非覆盖。所以三次后梯度为 4.0 → 8.0 → 12.0（每次加 4）。若需重置，用 `x.grad.zero_()`

## 块 11

## 场景 3：detach 冻结中间结果——从“梯度断流”理解 QLoRA 的冻结技巧

你在 QLoRA 微调中会遇到一个高频操作：**冻结部分参数**。QLoRA 的核心思路是冻结原模型权重，只训练低秩适配器。而 `detach()` 就是“冻结”的最小粒度工具——它把一个张量从计算图中“摘”出来，让它变成普通常数。

**先看正常情况：梯度完整回传**

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2          # y = 4

z1 = y * 3
z1.backward()
print(f"正常反向: x.grad = {x.grad}")   # 12.0
```

这里 `x → y → z1` 是一条完整链路，梯度从 `z1` 一路传回 `x`，所以 `x.grad = 12.0`（即 `dz1/dx = 6*x = 12`）。

**再看 detach：中间结果被“冻结”**

```python
x.grad.zero_()          # 先清零，避免累加
y_detached = y.detach() # 关键操作：y 变成常数
z2 = y_detached * 3
print(f"detach 后 z2.requires_grad: {z2.requires_grad}")  # False
# z2.backward()  # 会报错：RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
print(f"detach 后: x.grad = {x.grad}")  # 0.0  ← 梯度传不到 x 了！
```

**核心理解**：`y.detach()` 返回一个新张量，它**共享数据**但**不参与求导**。`z2` 的 `requires_grad` 是 `False`，所以调用 `backward()` 会直接报错——因为计算图里根本没有 `x` 这条路径了。梯度传不到 `x`，`x.grad` 保持 `0.0`。

**和 QLoRA 的对应关系**：QLoRA 中冻结原模型权重，本质上就是让这些权重的梯度“断流”——它们不参与反向传播，只有 LoRA

## 测评题目

### Q-KB0402-003
以下代码输出什么？
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 3
z.backward()
print(x.grad)
```

### Q-KB0402-001
PyTorch中，用于**反向传播**计算梯度的方法是？

### Q-KB0403-002
`nn.Linear(in_features=10, out_features=5)` 表示输入维度为________，输出维度为________。

### Q-KB0402-004
以下哪个操作会**断开**计算图，使后续操作不追踪梯度？

### Q-KB0405-001
以下哪个是**交叉熵损失**函数？

### Q-KB0401-001
以下哪个命令可以创建一个全为0的3x3张量？

### Q-KB0401-003
请将以下两个张量按**列方向**（dim=1）拼接：
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5], [6]])
```

### Q-KB0407-005
请写代码：从 `checkpoint.pth` 加载模型和优化器状态，并恢复训练。
