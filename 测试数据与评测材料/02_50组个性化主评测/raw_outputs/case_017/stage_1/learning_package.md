# 学习包

## 块 1

## 2.1 定义一个最简单的网络

你已经知道，`torch.nn` 是 PyTorch 的神经网络工具箱，而**所有网络模块都必须继承 `nn.Module`**。这个基类帮你自动管理参数、统一前向接口。现在，我们直接动手写一个最简单的网络，感受它的工作方式。

### 最小网络长什么样？

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)   # 一个全连接层：输入4维，输出2维

    def forward(self, x):
        return self.fc(x)
```

就这么简单。拆开看，你只做了两件事：

1. **在 `__init__` 里定义层**：`self.fc = nn.Linear(4, 2)` 创建了一个全连接层。`nn.Linear` 会自动生成权重和偏置，并注册为 `nn.Parameter`——这意味着它们默认 `requires_grad=True`，会被 `model.parameters()` 自动收集。
2. **在 `forward` 里写前向计算**：`return self.fc(x)` 定义了数据如何流过网络。你不需要手动调用 `self.fc.forward(x)`，PyTorch 会通过 `__call__` 机制自动执行 `forward`。

### 为什么必须写 `super().__init__()`？

这一行调用了父类 `nn.Module` 的初始化方法。**不写它，`nn.Module` 内部的参数管理机制就不会生效**，你的层不会被正确注册，`model.parameters()` 会返回空。这是新手最容易踩的坑。

### 怎么用这个网络？

```python
import torch

net = MyNet()                    # 实例化
x = torch.randn(3, 4)            # 3个样本，每个4维
out = net(x)                     # 直接调用实例，自动走 forward
print(out.shape)                 # 输出形状: torch.Size([3, 2])
```

注意：调用 `net(x)` 而不是 `net.forward(x)`。`nn.Module` 的 `__call__` 方法会先做一些内部处理（比如确保参数在正确设备上），再执行你的 `forward`。

### 参数去哪了？

```python
for name, param in net.named_parameters():
    print(name, param.shape, param.requires_grad)
```

输出：

```
fc.weight torch.Size([2, 4]) True
fc.bias torch.Size([2]) True
```

你什么都没做，权重和偏置就自动被收集了，而且 `requires_grad` 默认是 `True`。这就是 `nn.Module` 帮你省下的功夫——**你只管定义层和写前向逻辑，参数管理交给基类**。

### 本块学习检查

在 `pytorch` conda 环境中，完成以下操作：

1. 按上面的代码定义 `MyNet`，实例化后打印 `net`，观察输出中层的结构描述。
2. 调用

## 块 2

## 定义一个最简单的网络

你的目标是LoRA微调，而LoRA微调的核心就是“改造一个已有模型”。要改造模型，第一步是看懂模型是怎么搭出来的。下面这个 `SimpleNet` 就是最基础的样子，我们边看边拆。

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()          # 必须调用父类构造函数
        self.fc1 = nn.Linear(10, 20)   # 全连接层：输入10维，输出20维
        self.relu = nn.ReLU()          # 激活函数
        self.fc2 = nn.Linear(20, 1)    # 全连接层：输入20维，输出1维

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
x = torch.randn(4, 10)     # 4个样本，每个10维
output = model(x)          # 等价于 model.forward(x)，但推荐直接调用 model(x)
print(output.shape)        # torch.Size([4, 1])
```

**拆解三个关键点：**

1. **`super().__init__()` 必须写

## 块 3

## 3.1 全连接层：nn.Linear（①）

既然你的目标是 **LoRA 微调**，那全连接层就是你最先要打交道的“积木”。LoRA 的核心操作就是**冻结**原有全连接层的权重，在旁边加一个小矩阵去“模拟”权重的变化——所以理解 `nn.Linear` 的参数，是看懂 LoRA 代码的第一步。

**动手试一下（建议在 Jupyter 里跟着敲）：**

```python
fc = nn.Linear(in_features=784, out_features=128, bias=True)
```

这行代码创建了一个全连接层，它接收 784 个输入数字，输出 128 个数字。你可以把它想象成一个“压缩器”：把 784 维的信息浓缩成 128 维。

| 参数 | 含义 |
|---|---|
| `in_features` | 输入特征维度（本例：784）|
| `out_features` | 输出特征维度（本例：128）|
| `bias` | 是否使用偏置（默认 True）|

**快速验证（不引入新知识）：** 造一个形状为 `(1, 784)` 的随机张量喂给 `fc`

## 块 4

**目标：LoRA微调，先搞懂基础层**

你正在为LoRA微调打基础，这里快速过一遍PyTorch最常用的三个基础层。别纠结理论，直接看代码怎么用。

**激活函数层**——给网络加“非线性”的开关：
```python
nn.ReLU()           # 最常用：max(0, x)，负数直接砍掉
nn.Sigmoid()        # 输出压缩到0~1之间
nn.Tanh()           # 输出压缩到-1~1之间
nn.Softmax(dim=1)   # 多分类输出，dim=1表示在特征维度上算概率
```

**池化层**——缩小特征图尺寸，减少计算量

## 块 5

### 4.1 nn.Sequential：顺序堆叠（最常用）

**核心思想**：把网络层按顺序装进一个“管道”，数据自动从第一层流到最后一层，不用手动写前向传播。

**适合场景**：线性堆叠结构，比如你之后做LoRA微调时，替换或插入的线性层通常就是这种顺序关系。

**代码示例**（直接跑）：
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

x = torch.randn(4, 784)
output = model(x)      #

## 块 6

### 4.2 nn.ModuleList：列表式动态管理（③）

**核心要点**：当你想用循环遍历一组层时，`nn.ModuleList` 是你的好帮手。它就像一个“智能列表”，专门用来存放网络层。

**动手试试**：看下面这个动态网络，它把三个层装进一个列表里，前向传播时用 `for` 循环依次执行。

```python
class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ])

    def forward(self, x):
        for layer in self.layers

## 块 7

### 4.3 nn.ModuleDict：字典式命名访问（③）

做LoRA微调时，你常会遇到多分支网络——比如给不同层挂不同的适配器。`nn.ModuleDict` 就是为这种场景准备的：**用键名（字符串）来管理子模块**，像查字典一样按名字取层。

看这个例子：

```python
class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict({
            'branch_a': nn.Linear(10, 5),
            'branch_b': nn.Linear(10, 3),
            'branch_c': nn.Linear(10, 2)
        })

    def forward(self, x, branch_name):
        return self.branches[branch_name](x)
```

关键点：`forward()` 接收一个 `branch_name` 参数，然后用 `self.branches[branch_name]` 动态选择要执行的分支。**前向传播需要你手动指定用哪个键**，不像 `nn.Sequential` 那样自动串联。

### 4.4 三种容器速查（③）

| 容器

## 块 8

### 5.1 查看模型参数（①+②）

**动手做：先跑通，再看懂**

把下面代码复制到你的环境里跑一遍，观察输出：

```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

你会看到4行输出，对应4个可学习参数。注意两点：

1. **名字里的数字**：`0.weight` 和 `0.bias` 属于第一个 `Linear(10,5)`；`2.weight` 和 `2.bias` 属于第三个 `Linear(

## 块 9

**目标：LoRA微调前的“摸家底”**

你要做LoRA微调，第一步得知道模型长什么样。用 `model.modules()` 就能把模型里所有子模块（包括模型自身）递归地列出来：

```python
for idx, module in enumerate(model.modules()):
    print(f"模块 {idx}: {type(module).__name__}")
```

输出会像这样，从外到内一层层展示：

```
模块 0: Sequential
模块 1: Linear
模块 2: ReLU
模块 3: Linear
```

**动手验证**：在你准备微调的模型上跑这段代码，观察输出。数一数有几个 `Linear` 层——LoRA 通常只改这些层，记住它们的位置。

**顺带一提**：如果只想训练部分层（比如只训后面几层），

## 块 10

**本块学习检查**：运行下面两行代码，观察打印出的字典里键和值分别是什么类型？这能帮你确认 `state_dict()` 的结构。

```python
model = YourModel()  # 替换为你的模型
print(model.state_dict().keys())
print(model.state_dict()["权重参数名"].shape)
```

---

**保存模型参数**：训练完，用 `torch.save(model.state_dict(), "model.pth")` 把参数存成文件。`state_dict()` 返回一个字典，键是参数名，值是参数张量——这就是标准保存方式。

**加载模型参数**：先实例化一个**结构完全相同**的模型，再用 `model.load_state_dict(torch.load("model.pth"))`。注意：加载大模型时请确保显存充足（例如 Llama-2-7b FP16 需要至少14GB显存）。

# 模型许可证：请确认所用模型的开源许可证类型（如 Llama 2

## 块 11

### 6.1 L1Loss：平均绝对误差（MAE）

L1Loss 计算预测值与真实值之间**绝对差值的平均值**，公式为：`(|预测-真实|) 的平均`。

```python
criterion = nn.L1Loss()

input = torch.tensor([1.0, 3.0, 6.0])
target = torch.tensor([1.0, 4.0, 9.0])
loss = criterion(input, target)
print(loss)   # (|1-1| + |3-4| + |6-9|) / 3 = 4/3 ≈ 1.333
```

**适用场景**：回归问题，对异常值不敏感（即

## 块 12

## 6.3 CrossEntropyLoss：交叉熵损失（④）

做 LoRA 微调时，你几乎每次都会碰到这个损失函数——它是**多分类问题的标准选择**。好消息是：`CrossEntropyLoss` 内部已经帮你把 `LogSoftmax` 和 `NLLLoss` 打包好了，你只需要喂原始分数进去。

```python
criterion = nn.CrossEntropyLoss()

# input: (batch_size, num_classes) —— 原始 logits，不需要先 softmax
input = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
# target: (batch_size,) —— 类别索引
target = torch.tensor([0, 1])

loss = criterion(input, target)
print(loss)
```

**关键点（容易踩坑）**：
- 输入是**未经 softmax 的原始分数（logits）**，别手滑先做 softmax。
- 目标值是**类别索引**（如 `[0, 1]`），不是 one-hot 编码。
- 如果你用 `NLLLoss`，那输入必须是 `log_softmax` 得到的对数概率——但用 `CrossEntropyLoss` 就省了这一步。

## 6.4 三种损失函数速查（④）

| 损失函数 | 公式特点 | 适用任务 | 输入要求 |
|---|---|---|---|
| `L1

## 块 13

**目标**：为LoRA微调打基础——先搞懂PyTorch默认初始化，以及为什么需要Xavier。

**动手试**：创建一个小网络，打印`nn.Linear`的权重，观察数值范围。默认初始化因层类型而异，例如 `nn.Linear` 和 `nn.Conv2d` 默认使用 `kaiming_uniform_`，而某些层可能使用其他分布。

**关键点**：深层网络中，默认初始化可能不够理想。信号在层层传递中会衰减或爆炸。

**解决方案：Xavier初始化（Glorot）**

核心思想：让每层输出的方差与输入方差保持一致，避免信号衰减或爆炸。

适用场景：Sigmoid、Tanh等饱和型激活函数。

**代码实操**（在模型定义后添加）：

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)   # Xavier 均匀分布
        # 或 nn.init.xavier

## 块 14

**Kaiming / HE 初始化：让 ReLU 网络不“熄火”**

你在为 LoRA 微调做准备，先掌握这个基础：ReLU 会把一半负值截断成 0，若初始化不当，梯度容易消失。Kaiming 初始化就是为此设计的——它调整方差，抵消 ReLU 的“截断”效应，让信号稳定传播。

**适用**：ReLU、LeakyReLU 这类非饱和激活函数。

**PyTorch 实操**（在模型定义后、训练前执行）：

```python
for m in model.modules():
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='

## 块 15

## 7.4 初始化策略速查

你正在为 LoRA 微调打基础，初始化是理解模型训练的第一步。记住这张表，实战时直接查：

| 方法 | 适用场景 | PyTorch 函数 |
|---|---|---|
| 默认初始化 | 未指定初始化方式时（不同层类型有各自的默认方法） | 无需操作 |
| Xavier/Glorot | 使用 Sigmoid、Tanh 等饱和激活函数的网络 | `nn.init.xavier_uniform_` / `xavier_normal_` |
| Kaiming/HE | 使用 ReLU 及其变体激活的网络 | `nn.init.kaiming_uniform_` / `kaiming_normal_` |
| 常数初始化 | 偏置项初始化为 0 | `nn.init.constant_(m.bias, 0)`

## 块 16

## 搭建 CIFAR-10 图像分类的简单 CNN

你正在为 LoRA 微调打基础，而理解一个 CNN 的完整结构——从特征提取到分类输出——是后续做参数高效微调的前提。下面我们直接动手，把这个模型拆开看。

### 一、整体结构

这个 `SimpleCNN` 由两部分组成：

1. **特征提取层（`self.features`）**：用卷积层从图像中提取特征
2. **分类器（`self.classifier`）**：把提取到的特征映射到 10 个类别

### 二、特征提取层

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 3→16通道
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),         # 尺寸减半

    nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16→32通道
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)          # 尺寸再减半
)
```

**关键点**：
- 输入是 3 通道的 32×32 图像（CIFAR-10 的格式）
- 第一层卷积把通道数从 3 扩到 16，`padding=1` 保持空间尺寸不变
- 每次 `MaxPool2d` 让尺寸减半：32→16→8
- 最终特征图尺寸是 32 通道 × 8×8

### 三、分类器

```python
self.classifier = nn.Sequential(
    nn.Flatten(),                                  # 展平
    nn.Linear(32 * 8 * 8, 128),                   # 展平后 2048 维 → 128 维
    nn.ReLU(),
    nn.Linear(128, 10)                             # 输出 10 个类别
)
```

**关键点**：
- `Flatten()` 把 32×8×8 的特征图展平成 2048 维向量
- 第一个全连接层压缩到 128 维
- 第二个全连接层输出 10 个值，对应 10 个类别

### 四、参数初始化

```

## 块 17

## 九、常见问题与解决（①+②+③+④+⑤）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 模型参数不被优化器更新 | 层没有用 `self.xxx =` 定义 | 确保在 `__init__` 中使用 `self.layer = nn.Linear(...)` |
| `Sequential` 里的层无法单独访问 | 不知道索引规则 | `model[0]` 访问第一层，`model[2]` 访问第三层 |
| `CrossEntropyLoss` 报错 | 输入已经过 softmax | `CrossEntropyLoss` 内部已包含 LogSoftmax 和 NLLLoss，输入原始 logits 即可，不要额外进行 softmax 或 log_softmax |
| 损失值不下降 / 模型不收敛 | 参数初始化不当或学习率问题 | 尝试 Kaiming/Xavier 初始化，调小学习率 |
| 深层网络梯度消失 | Sigmoid/Tanh + 默认初始化 | 换 ReLU + Kaiming 初始化，或加 BatchNorm |
| 冻结层后优化器报错 | 优化器接收了全部参数 | 确保冻结层的 `requires_grad` 已设为 False，并使用 `filter(lambda p: p.requires_grad, model.parameters())` 过滤参数 |
| `load_state_dict` 报错 `Missing key` | 保存和加载的模型结构不一致 | 确保加载前实例化的模型结构与保存时完全一致 |

---

## 十、总结与后续学习路线（①+②+⑤）

恭喜你完成了 torch.nn 模块章节的学习！你现在应该掌握了：

- ✅ `nn.Module` 基类：继承、定义 `__init__` 和 `forward`
- ✅ 常用层：`nn.Linear`、`nn.Conv2d`、`nn.ReLU`、`nn.MaxPool2d`、`nn.Flatten`
- ✅ 三种容器：`nn.Sequential`（顺序自动）、`nn.ModuleList`（列表动态）、`nn.ModuleDict`（字典命名）
- ✅ 参数管理：`parameters()`、`named_parameters()`、`modules()`、冻结层、`state_dict()`
- ✅ 三种核心损失函数：`L1Loss

## 测评题目

### Q-KB0403-004
`nn.Module` 的子类中，`forward` 方法的作用是？

### Q-KB0403-001
以下哪个是PyTorch中定义**线性层**（全连接层）的类？

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB0402-005
请写代码：计算 `y = x1 * x2 + x3` 对 `x1`、`x2`、`x3` 的梯度，其中 `x1=2, x2=3, x3=4`。

### Q-KB0402-001
PyTorch中，用于**反向传播**计算梯度的方法是？

### Q-KB0403-005
请写代码：使用 `nn.Sequential` 定义一个包含 `Linear(10, 5)`、`ReLU`、`Linear(5, 1)` 的网络。

### Q-KB0401-005
给定张量 `x` 形状为 `(2, 3, 4)`，请写出将其转置为 `(4, 3, 2)` 的代码。

### Q-KB0406-004
`model.train()` 和 `model.eval()` 的主要区别是？
