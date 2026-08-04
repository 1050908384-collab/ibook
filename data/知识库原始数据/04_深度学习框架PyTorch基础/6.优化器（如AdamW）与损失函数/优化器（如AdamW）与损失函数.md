# 优化器（如 AdamW）与损失函数

① 底稿：《Pytorch学习笔记--损失函数以及优化器的使用》\
　　网址：https://blog.csdn.net/aaaax__/article/details/142783541

② 参考文章：《PyTorch优化器进阶：SGD/AdamW/Adafactor…8种优化器对比+适用场景总结》\
　　网址：https://blog.csdn.net/2501_91798322/article/details/151368227

③ 参考文章：《【深度学习】AdamW优化器如何通过解耦权重衰减提升模型泛化能力？》\
　　网址：https://blog.csdn.net/weixin_29074295/article/details/158709306

④ 参考文章：《图像分类优化器选型实战：从SGD到LAMB的工程解剖》\
　　网址：https://blog.csdn.net/weixin_30580341/article/details/94950674

⑤ 参考文章：《PyTorch优化器完全指南》\
　　网址：https://blog.csdn.net/qq_34941290/article/details/156203517

⑥ 参考文章：《详细介绍：PyTorch 损失函数与优化器全面指南：从理论到实践》\
　　网址：https://www.cnblogs.com/yjbjingcha/p/19078066

> **适合人群**：已完成"Dataset/DataLoader 数据装载"章节，能够使用 `DataLoader` 批量加载数据、使用 `nn.Module` 搭建简单神经网络的初学者。本节时长约 1 小时，目标是理解损失函数与优化器的核心作用，掌握 CrossEntropyLoss/MSELoss 的适用场景，能够配置 SGD、Adam 和 AdamW 优化器，并理解学习率调度器的基本用法，为下一节编写完整训练循环做好准备。（①+⑥）
>
> **前置说明**：上节我们解决了"数据怎么来"的问题，本节解决"模型怎么学"的问题。损失函数告诉模型"错在哪里"，优化器告诉模型"怎么改"。所有代码请在已安装 PyTorch 的 `pytorch` conda 环境中运行。

## 一、损失函数与优化器：模型训练的"裁判"和"教练"（①+⑥）

神经网络训练的本质是一个**不断减小误差**的过程：

1.  **损失函数（Loss Function）**：衡量模型预测值与真实值之间的差距，相当于"裁判打分"。（①+⑥）
2.  **优化器（Optimizer）**：根据损失函数的梯度，调整模型参数以减小损失，相当于"教练指导"。（⑤+⑥）

> **官方文档**：\
> PyTorch 损失函数完整列表 https://pytorch.org/docs/stable/nn.html#loss-functions\
> PyTorch 优化器完整列表 https://pytorch.org/docs/stable/optim.html

## 二、常用损失函数（①+⑥）

> 在"torch.nn 模块"章节中，我们已经初步认识了 `L1Loss`、`MSELoss` 和 `CrossEntropyLoss`。本节在此基础上，更深入地讲解它们的内部机制和参数细节。

### 2.1 L1Loss：平均绝对误差（MAE）（①+⑥）

`criterion = nn.L1Loss()`

计算预测值与目标值之间绝对差值的平均。对异常值不敏感，适合回归问题。（①）

### 2.2 MSELoss：均方误差（①+⑥）

`criterion = nn.MSELoss(reduction='mean')`

计算平方差值的平均。对大误差惩罚更重（平方放大），是回归问题的默认选择。（①+⑥）

`reduction` 参数控制输出方式：（⑥）

-   `'mean'`（默认）：返回平均值
-   `'sum'`：返回总和
-   `'none'`：不聚合，返回元素级损失（形状与输入相同）

### 2.3 CrossEntropyLoss：交叉熵损失（①+⑥）

**多分类问题的标准损失函数**，内部已经合并了 `LogSoftmax` 和 `NLLLoss` 两个操作。（⑥）

`criterion = nn.CrossEntropyLoss()`

> **重要输入要求**：（①+⑥）
>
> -   **输入**：`(N, C)` 的**原始 logits**（未经 softmax 的分数）
> -   **目标**：`(N,)` 的类别索引（`0 ≤ target[i] ≤ C-1`）

```python
import torch
import torch.nn as nn

# 3 个样本，5 个类别
outputs = torch.randn(3, 5)          # 原始 logits
labels = torch.tensor([1, 0, 4])     # 类别索引

criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
print(loss)   # 标量张量
```

> **内部机制**：`CrossEntropyLoss` 等价于 `nn.LogSoftmax(dim=1) + nn.NLLLoss()`。它直接在数值上计算 `-x[class] + log(sum(exp(x[j])))`，避免了先 softmax 再取 log 可能带来的数值不稳定问题。（⑥）

**类别不平衡时**，可通过 `weight` 参数给不同类别设置权重：（⑥）

```python
# 假设类别 1 的样本很少，给它更高的权重
weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=weights)
```

### 2.4 三种损失函数速查（①+⑥）

| 损失函数 | 适用任务 | 输入要求 | 特点 |
|---|---|---|---|
| `L1Loss` | 回归 | 预测值与目标值形状相同 | 对异常值不敏感 |
| `MSELoss` | 回归 | 预测值与目标值形状相同 | 对大误差惩罚重 |
| `CrossEntropyLoss` | 多分类 | logits `(N, C)` + 索引 `(N,)` | 内部含 softmax，数值稳定 |

## 三、优化器：从 SGD 到 AdamW（⑤）

优化器决定了模型参数如何根据梯度进行更新。本节介绍三种最常用优化器：SGD（带动量）、Adam 和 AdamW。（⑤）

### 3.1 SGD with Momentum：经典但有效（⑤）

SGD（随机梯度下降）是最基础的优化器。加入 Momentum（动量）后，优化器会"记住"之前的更新方向，像滚雪球一样加速收敛并减少震荡。（⑤）

**核心参数**：

```python
from torch.optim import SGD

optimizer = SGD(
    model.parameters(),
    lr=0.01,          # 学习率
    momentum=0.9,     # 动量系数（0.9 是常用值）
    weight_decay=1e-4 # 权重衰减（L2 正则化）
)
```

-   `lr`：学习率，控制每次更新的步长。SGD 对学习率敏感，通常需要配合学习率调度器使用。
-   `momentum`：动量系数。设为 0 表示纯 SGD，设为 0.9 表示带动量。
-   `weight_decay`：L2 正则化系数，防止过拟合。（⑤）

> **适用场景**：图像分类等计算机视觉任务中，SGD+Momentum 往往能获得较好的泛化能力。缺点是收敛速度较慢，需要精细调参。（⑤）

### 3.2 Adam：自适应学习率（⑤）

Adam（Adaptive Moment Estimation）融合了动量法和 RMSprop 的优点，为每个参数单独调整学习率，收敛速度快且对学习率不敏感。（⑤）

```python
from torch.optim import Adam

optimizer = Adam(
    model.parameters(),
    lr=0.001,            # 学习率（通常比 SGD 小一个数量级）
    betas=(0.9, 0.999),  # 一阶矩和二阶矩衰减率
    eps=1e-8,            # 数值稳定性常数
    weight_decay=0       # 权重衰减（注意：Adam 中此参数有坑！）
)
```

-   `betas=(0.9, 0.999)`：一阶矩（动量）和二阶矩（自适应学习率）的衰减率，默认值适用于大多数场景。
-   `eps`：防止除零的小常数，一般不需要修改。（⑤）

> **Adam 的"坑"**：Adam 的 `weight_decay` 实现与 SGD 不同。在 Adam 中，权重衰减项会被二阶矩 `v_t` 缩放，导致正则化效果不稳定——某些参数被过度抑制，某些又抑制不足。（③+④）

### 3.3 AdamW：解耦权重衰减（重点）（③+⑤）

AdamW 是 Adam 的改进版，核心改动只有一点：**将权重衰减与梯度更新彻底解耦**。（③+⑤）

#### 通俗理解（③）

把训练模型比作"培养运动员"：

-   **梯度更新** = 根据训练表现调整营养配比（让运动员跑得更快）
-   **权重衰减** = 控制饮食防止过度肥胖（防止过拟合）

Adam 的问题是：它把"减肥计划"混进了"营养配比"里，由智能手环（自适应学习率）统一调配。结果手环看到今天训练量大（梯度大），就自作主张放松减肥计划，导致减肥力度不稳定。（③）

AdamW 的解决方案：**吃饭归吃饭，减肥归减肥**。先按 Adam 的规则更新参数，再单独、固定地执行权重衰减，互不干扰。（③）

#### 数学对比（⑤）

-   **原始 Adam**（带 weight_decay）：参数更新项中包含被自适应学习率缩放过的权重衰减
-   **AdamW**：权重衰减独立于自适应更新，直接对参数进行衰减

#### PyTorch 用法（⑤）

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01   # 推荐设为 0.01，正则效果比 Adam 更稳定
)
```

> **实践建议**：在 Transformer 类模型（如 BERT、GPT）和现代深度学习任务中，**AdamW 已基本取代 Adam 成为默认选择**。相同任务下，AdamW 的泛化能力通常优于 Adam，且无需额外调参。（②+③）

### 3.4 三种优化器速查（⑤）

| 优化器 | 核心特点 | 学习率建议 | 适用场景 |
|---|---|---|---|
| `SGD` + Momentum | 固定学习率 + 动量 | 0.01 ~ 0.1 | 图像分类，追求泛化 |
| `Adam` | 自适应学习率 | 0.001 | 快速原型验证（建议改用 AdamW） |
| `AdamW` | 自适应 + 解耦权重衰减 | 0.001 | Transformer、大多数现代任务 |

## 四、学习率调度器：让学习率"动"起来（⑥）

固定学习率往往不够高效：初期需要较大学习率快速接近最优解，后期需要较小学习率精细调整。学习率调度器（LR Scheduler）就是用来动态调整学习率的工具。（⑥）

### 4.1 StepLR：阶梯式衰减（⑥）

每隔固定步数，学习率乘以衰减系数：

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 每 30 个 epoch，学习率 × 0.1
```

### 4.2 CosineAnnealingLR：余弦退火（⑥）

学习率按余弦曲线从初始值平滑下降到接近 0，现代训练常用：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# T_max：半个余弦周期的 epoch 数
# eta_min：学习率下限
```

### 4.3 使用方式（⑥）

调度器在**每个 epoch 结束后**调用 `step()`：

```python
for epoch in range(num_epochs):
    for data, target in train_loader:
        # ... 训练步骤 ...
        optimizer.step()
    scheduler.step()   # ← 每个 epoch 结束后更新学习率
```

## 五、完整训练循环：损失函数 + 优化器 + 调度器（①+⑤+⑥）

把前面学的知识串起来，一个完整的训练步骤长这样：（①+⑥）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========== 1. 准备数据 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# ========== 2. 定义模型（简单 CNN） ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleCNN()

# ========== 3. 损失函数 ==========
criterion = nn.CrossEntropyLoss()

# ========== 4. 优化器（AdamW，推荐） ==========
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# ========== 5. 学习率调度器（余弦退火） ==========
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# ========== 6. 训练循环 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()   # 1. 清零旧梯度
        loss.backward()         # 2. 计算新梯度
        optimizer.step()        # 3. 更新参数

        running_loss += loss.item()

    # 每个 epoch 结束后更新学习率
    scheduler.step()

    avg_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

print("训练完成！")
```

**训练循环四步口诀**：前向 → 清零 → 反向 → 更新。（①）

## 六、常见问题与解决（①+③+⑤+⑥）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 损失值不下降 | 学习率过大或过小 | 尝试 `lr=0.001`（Adam/AdamW）或 `lr=0.01`（SGD） |
| 损失值震荡剧烈 | 学习率过大 | 减小学习率，或增大 batch_size |
| 验证集精度远低于训练集 | 过拟合 | 增大 `weight_decay`（如 0.01→0.05），或换 SGD |
| Adam 泛化能力不如 SGD | Adam 的 weight_decay 被缩放 | **改用 AdamW**，保持 `weight_decay=0.01` |
| 训练后期损失停滞 | 学习率没有衰减 | 添加 `CosineAnnealingLR` 或 `StepLR` |
| `CrossEntropyLoss` 报错 | 输入已做 softmax | 输入原始 logits，不要先过 softmax |
| 忘记 `zero_grad()` | 梯度累加 | 每次 `step()` 前必须调用 `optimizer.zero_grad()` |
| MSELoss 输出形状不对 | `reduction` 参数设置不当 | 默认 `'mean'` 输出标量，`'none'` 输出元素级 |

## 七、总结与后续学习路线（①+③+⑤）

恭喜你完成了优化器与损失函数章节的学习！你现在应该掌握了：

-   ✅ 三种核心损失函数：`L1Loss`、`MSELoss`（`reduction` 参数）、`CrossEntropyLoss`（内部机制、类别加权）
-   ✅ SGD with Momentum：动量原理、适用场景、调参要点
-   ✅ Adam：自适应学习率、`betas` 和 `eps` 参数含义
-   ✅ **AdamW**：解耦权重衰减的核心原理、为什么取代 Adam、推荐参数
-   ✅ 学习率调度器：`StepLR` 和 `CosineAnnealingLR` 的用法
-   ✅ 完整的训练循环：损失函数 + 优化器 + 调度器协同工作

> **下一步预告**：在下一节 **"训练循环编写"** 中，我们将把本节所学与前面的 `nn.Module`、`Dataset/DataLoader` 彻底整合，编写一个包含**训练 + 验证 + 指标监控**的完整训练脚本。你会学习如何计算准确率、如何保存最优模型、如何绘制损失曲线。现在，试着把上面的 CIFAR-10 示例跑起来，并分别用 SGD 和 AdamW 对比训练效果吧！