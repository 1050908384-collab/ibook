# torch.nn 模块

① 底稿：《【pytorch】模型的定义、参数查看和训练；卷积层计算规则；模型修改的小技巧》\
　　网址：https://blog.csdn.net/weixin_69722030/article/details/127548146

② 参考文章：《解析 nn.Module 的 parameters 与 modules》\
　　网址：https://blog.csdn.net/sweet_ran/article/details/151875198

③ 参考文章：《nn.ModuleList()、nn.Sequential() 和 nn.ModuleDict() 区别》\
　　网址：https://blog.csdn.net/KIKI3666/article/details/141168046

④ 参考文章：《Pytorch框架04-损失函数(L1Loss()/MSELoss()/CrossEntropyLoss())》\
　　网址：https://blog.csdn.net/XYX_888/article/details/145833595

⑤ 参考文章：《pytorch-13.3 模型参数初始化方法 Xavier和Kaiming（HE）》\
　　网址：https://blog.csdn.net/weixin_45311418/article/details/139185471

> **适合人群**：已完成"自动求导机制"章节，理解 `requires_grad`、`backward()` 和梯度计算图的初学者。本节时长约 1.5 小时，目标是掌握使用 `torch.nn` 构建神经网络的核心方法，包括定义模型、使用常用层、组织网络结构、管理参数、选择损失函数和初始化参数，能够独立搭建一个简单的多层神经网络。（①+②）
>
> **前置说明**：上节我们学过，PyTorch 的自动求导引擎会为 `requires_grad=True` 的张量计算梯度。`torch.nn` 中的所有层参数默认就是 `requires_grad=True` 的叶子节点，因此调用 `loss.backward()` 后，模型参数会自动获得梯度，供优化器更新。本节所有代码请在 `pytorch` conda 环境中运行。

---

## 一、为什么需要 torch.nn？（①）

上节我们用原始张量运算手动实现了一个线性模型：定义权重张量、写前向公式、手动计算损失、手动更新参数。这种方式在简单场景下可行，但当网络层数增加、结构变复杂时，代码会变得混乱且难以维护。（①）

`torch.nn` 是 PyTorch 提供的**神经网络构建工具箱**，它帮你做了三件事：（①+②）

1. **封装常用层**：`nn.Linear`、`nn.Conv2d`、`nn.ReLU` 等，开箱即用
2. **自动管理参数**：层的权重和偏置自动注册为 `nn.Parameter`，自动被 `model.parameters()` 收集
3. **统一前向接口**：所有模块都继承自 `nn.Module`，只需实现 `forward()` 方法

> **官方文档**：PyTorch nn 模块完整 API 参考 https://pytorch.org/docs/stable/nn.html

---

## 二、nn.Module：所有网络的基类（①+②）

在 PyTorch 中，**所有神经网络模块都必须继承 `nn.Module`**。它提供了参数管理、设备转移、状态保存等基础能力。（①+②）

### 2.1 定义一个最简单的网络

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
        # 定义前向传播流程
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleNet()

# 测试前向传播
x = torch.randn(4, 10)     # 4个样本，每个10维
output = model(x)          # 等价于 model.forward(x)，但推荐直接调用 model(x)
print(output.shape)        # torch.Size([4, 1])
```

> **重要规则**：`__init__()` 里必须用 `self.xxx = nn.XXX()` 的方式定义层，这样 PyTorch 才能自动识别并管理这些层的参数。如果只是普通变量 `xxx = nn.Linear(...)`，PyTorch 不会把它当作模型的一部分。（②）

---

## 三、常用神经网络层（①）

### 3.1 全连接层：nn.Linear（①）

```python
fc = nn.Linear(in_features=784, out_features=128, bias=True)
```

| 参数 | 含义 |
|---|---|
| `in_features` | 输入特征维度 |
| `out_features` | 输出特征维度 |
| `bias` | 是否使用偏置（默认 True） |

### 3.2 卷积层：nn.Conv2d（①）

```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

| 参数 | 含义 |
|---|---|
| `in_channels` | 输入通道数（如 RGB 图像为 3）|
| `out_channels` | 输出通道数（卷积核个数）|
| `kernel_size` | 卷积核大小（3 表示 3×3）|
| `stride` | 步长（默认 1）|
| `padding` | 边缘填充（默认 0）|

> 输出尺寸公式：$H_{out} = \lfloor \frac{H_{in} + 2 \times padding - kernel\_size}{stride} + 1 \rfloor$（①）

### 3.3 激活函数层（①）

```python
nn.ReLU()           # 最常用：max(0, x)
nn.Sigmoid()        # 1 / (1 + exp(-x))
nn.Tanh()           # 双曲正切
nn.Softmax(dim=1)   # 多分类输出，dim 指定在哪个维度上做 softmax
```

### 3.4 池化层（①）

```python
nn.MaxPool2d(kernel_size=2, stride=2)   # 最大池化：取 2×2 区域内的最大值
nn.AvgPool2d(kernel_size=2, stride=2)   # 平均池化：取 2×2 区域内的平均值
```

### 3.5 展平层（①）

```python
nn.Flatten()        # 将多维张量展平为一维，常用于卷积层后接全连接层前
```

---

## 四、容器：组织网络结构的三种方式（①+③）

当网络层数增多时，需要容器来组织和管理子模块。PyTorch 提供了三种容器：（③）

### 4.1 nn.Sequential：顺序堆叠（最常用）（①+③）

适用于**线性堆叠**的网络结构，数据按添加顺序自动流经每一层。（③）

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

x = torch.randn(4, 784)
output = model(x)      # 自动按顺序前向传播
print(output.shape)    # torch.Size([4, 10])
```

> `nn.Sequential` 内部已经实现了 `forward()`，你不需要再写。（③）

### 4.2 nn.ModuleList：列表式动态管理（③）

适用于需要**动态选择或遍历**层的情况，比如循环遍历不同层。（③）

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
        for layer in self.layers:
            x = layer(x)
        return x
```

> 与 Python 列表 `[]` 的区别：`nn.ModuleList` 中的模块会被自动注册为模型参数，能被优化器找到。（③）

### 4.3 nn.ModuleDict：字典式命名访问（③）

适用于需要**按名称选择**层的情况，比如多分支网络。（③）

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

### 4.4 三种容器速查（③）

| 容器 | 访问方式 | 前向传播 | 适用场景 |
|---|---|---|---|
| `nn.Sequential` | 索引 | 自动 | 简单的线性堆叠网络 |
| `nn.ModuleList` | 索引 | 需在 `forward()` 中手动遍历 | 动态层数、循环结构 |
| `nn.ModuleDict` | 键名 | 需在 `forward()` 中手动指定 | 多分支、条件选择 |

---

## 五、参数管理：查看与遍历（①+②）

### 5.1 查看模型参数（①+②）

```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 查看所有可学习参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# 输出：
# 0.weight: torch.Size([5, 10])
# 0.bias: torch.Size([5])
# 2.weight: torch.Size([2, 5])
# 2.bias: torch.Size([2])
```

> `named_parameters()` 返回 `(参数名, 参数张量)` 的迭代器。`parameters()` 只返回参数张量。（②）

### 5.2 查看模型结构（②）

```python
# 查看所有子模块（包括自身，递归遍历）
for idx, module in enumerate(model.modules()):
    print(f"模块 {idx}: {type(module).__name__}")
# 输出：
# 模块 0: Sequential
# 模块 1: Linear
# 模块 2: ReLU
# 模块 3: Linear
```

### 5.3 冻结部分层（②）

有时你希望只训练网络的一部分（如迁移学习），可以冻结某些层的参数：（②）

```python
# 冻结第一层
for param in model[0].parameters():
    param.requires_grad = False

# 验证：优化器只会更新 requires_grad=True 的参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 5.4 保存与加载模型状态（①）

```python
# 保存模型参数
torch.save(model.state_dict(), "model.pth")

# 加载模型参数（必须先实例化相同结构的模型）
model.load_state_dict(torch.load("model.pth"))
```

> `state_dict()` 返回一个字典，键是参数名，值是参数张量。这是保存和加载模型的标准方式。（①+②）

---

## 六、损失函数（④）

损失函数衡量模型预测值与真实值之间的差距，是训练优化的目标。`torch.nn` 提供了丰富的内置损失函数。（④）

> 下节"优化器与损失函数"会深入讲解损失函数的选择策略，本节先介绍最常用的三种。

### 6.1 L1Loss：平均绝对误差（MAE）（④）

```python
criterion = nn.L1Loss()

input = torch.tensor([1.0, 3.0, 6.0])
target = torch.tensor([1.0, 4.0, 9.0])
loss = criterion(input, target)
print(loss)   # (|1-1| + |3-4| + |6-9|) / 3 = 4/3 ≈ 1.333
```

适用场景：回归问题，对异常值不敏感。（④）

### 6.2 MSELoss：均方误差（④）

```python
criterion = nn.MSELoss()

loss = criterion(input, target)
print(loss)   # [(1-1)^2 + (3-4)^2 + (6-9)^2] / 3 = 10/3 ≈ 3.333
```

适用场景：回归问题，对较大误差惩罚更重（平方放大）。（④）

### 6.3 CrossEntropyLoss：交叉熵损失（④）

**多分类问题的标准损失函数**，内部已经包含了 `Softmax` 和 `NLLLoss`。（④）

```python
criterion = nn.CrossEntropyLoss()

# input: (batch_size, num_classes) —— 原始 logits，不需要先 softmax
input = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
# target: (batch_size,) —— 类别索引
target = torch.tensor([0, 1])

loss = criterion(input, target)
print(loss)
```

> **重要**：`CrossEntropyLoss` 的输入是**未经 softmax 的原始分数（logits）**，目标值是**类别索引**（不是 one-hot）。不要先做 softmax；若使用 `NLLLoss`，输入必须是 `log_softmax` 得到的对数概率。（④）

### 6.4 三种损失函数速查（④）

| 损失函数 | 公式特点 | 适用任务 | 输入要求 |
|---|---|---|---|
| `L1Loss` | 绝对值之和 | 回归 | 预测值与目标值形状相同 |
| `MSELoss` | 平方差之和 | 回归 | 预测值与目标值形状相同 |
| `CrossEntropyLoss` | 负对数似然 | 多分类 | logits `(N, C)` + 索引 `(N,)` |

---

## 七、参数初始化（⑤）

神经网络训练开始前，参数初始值的选择会影响收敛速度和最终效果。不合理的初始化可能导致梯度消失或梯度爆炸。（⑤）

### 7.1 默认初始化

PyTorch 的 `nn.Linear` 和 `nn.Conv2d` 等层在创建时会自动初始化参数，通常采用均匀分布或正态分布。但在深层网络中，默认初始化可能不够理想。（⑤）

### 7.2 Xavier 初始化（Glorot）（⑤）

**核心思想**：让每一层输出的方差与输入的方差保持一致，避免信号在深层网络中衰减或爆炸。（⑤）

- **适用激活函数**：Sigmoid、Tanh 等饱和型激活函数
- **PyTorch 实现**：

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)   # Xavier 均匀分布
        # 或 nn.init.xavier_normal_(m.weight)  # Xavier 正态分布
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
```

### 7.3 Kaiming / HE 初始化（⑤）

**核心思想**：针对 ReLU 及其变体（LeakyReLU）设计，考虑 ReLU 会"截断"一半负值的特点，调整初始化方差。（⑤）

- **适用激活函数**：ReLU、LeakyReLU 等非饱和型激活函数
- **PyTorch 实现**：

```python
for m in model.modules():
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # 或 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
```

### 7.4 初始化策略速查（⑤）

| 方法 | 适用场景 | PyTorch 函数 |
|---|---|---|
| 默认初始化 | 一般浅层网络 | 无需操作 |
| Xavier/Glorot | Sigmoid、Tanh 激活的深层网络 | `nn.init.xavier_uniform_` / `xavier_normal_` |
| Kaiming/HE | ReLU 激活的深层网络 | `nn.init.kaiming_uniform_` / `kaiming_normal_` |
| 常数初始化 | 偏置项初始化为 0 | `nn.init.constant_(m.bias, 0)` |

> **实践建议**：现代框架中，使用 ReLU 的深层网络优先选 Kaiming 初始化；使用 Sigmoid/Tanh 的优先选 Xavier。BatchNorm 层可以在一定程度上缓解初始化不当的问题。（⑤）

---

## 八、综合实战演练（①+②+③+④）

请在你的 Python 环境中完成以下操作：

### 场景：搭建一个用于 CIFAR-10 图像分类的简单 CNN

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 3→16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 尺寸减半

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16→32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)          # 尺寸再减半
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # 展平
            nn.Linear(32 * 8 * 8, 128),                   # 假设输入 32×32，经过两次池化后 8×8
            nn.ReLU(),
            nn.Linear(128, num_classes)                    # 输出 10 个类别
        )

        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# 实例化并测试
model = SimpleCNN(num_classes=10)

# 查看模型结构
print(model)

# 查看参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

# 模拟一次前向传播
x = torch.randn(4, 3, 32, 32)    # batch=4, 3通道, 32×32图像
output = model(x)
print(f"输出形状: {output.shape}")  # torch.Size([4, 10])

# 模拟损失计算
criterion = nn.CrossEntropyLoss()
target = torch.randint(0, 10, (4,))   # 4个随机类别标签
loss = criterion(output, target)
print(f"损失值: {loss.item():.4f}")

# 反向传播
loss.backward()

# 查看某个参数的梯度
print(f"fc1 weight grad shape: {model.classifier[1].weight.grad.shape}")
```

---

## 九、常见问题与解决（①+②+③+④+⑤）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 模型参数不被优化器更新 | 层没有用 `self.xxx =` 定义 | 确保在 `__init__` 中使用 `self.layer = nn.Linear(...)` |
| `Sequential` 里的层无法单独访问 | 不知道索引规则 | `model[0]` 访问第一层，`model[2]` 访问第三层 |
| `CrossEntropyLoss` 报错 | 输入已经过 softmax | `CrossEntropyLoss` 内部已含 softmax，输入原始 logits 即可 |
| 损失值不下降 / 模型不收敛 | 参数初始化不当或学习率问题 | 尝试 Kaiming/Xavier 初始化，调小学习率 |
| 深层网络梯度消失 | Sigmoid/Tanh + 默认初始化 | 换 ReLU + Kaiming 初始化，或加 BatchNorm |
| 冻结层后优化器报错 | 优化器接收了全部参数 | 用 `filter(lambda p: p.requires_grad, model.parameters())` |
| `load_state_dict` 报错 `Missing key` | 保存和加载的模型结构不一致 | 确保加载前实例化的模型结构与保存时完全一致 |

---

## 十、总结与后续学习路线（①+②+⑤）

恭喜你完成了 torch.nn 模块章节的学习！你现在应该掌握了：

- ✅ `nn.Module` 基类：继承、定义 `__init__` 和 `forward`
- ✅ 常用层：`nn.Linear`、`nn.Conv2d`、`nn.ReLU`、`nn.MaxPool2d`、`nn.Flatten`
- ✅ 三种容器：`nn.Sequential`（顺序自动）、`nn.ModuleList`（列表动态）、`nn.ModuleDict`（字典命名）
- ✅ 参数管理：`parameters()`、`named_parameters()`、`modules()`、冻结层、`state_dict()`
- ✅ 三种核心损失函数：`L1Loss`、`MSELoss`、`CrossEntropyLoss`
- ✅ 参数初始化：Xavier（Sigmoid/Tanh）和 Kaiming/HE（ReLU）
- ✅ 搭建了一个完整的 CNN 模型并完成了前向传播和反向传播

> **下一步预告**：在下一节 **"Dataset/DataLoader 数据装载"** 中，我们将解决"如何把原始数据喂给模型"的问题。你将学习 `torch.utils.data.Dataset` 和 `DataLoader`，掌握批量加载、数据打乱和多进程读取。随后还将进入 **"优化器与损失函数"**，深入学习 SGD、Adam、AdamW 等优化器的工作原理和调参技巧。现在，试着用 `nn.Module` 搭建一个你自己的神经网络吧！
