# 学习包

## 块 1

**本块学习检查**  
在开始前，请先完成一个最小示例：用 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 配置设备，并打印当前设备。确认你能解释这行代码的作用（自动选择 GPU 或 CPU），再继续本块。

---

## 3.1 设备配置：让代码自动选择运行环境

在训练循环中，**第一步是确定代码跑在哪里**——GPU 还是 CPU。这决定了训练速度，也影响后续所有张量（tensor）的存放位置。

### 为什么需要设备配置？

PyTorch 的计算默认在 CPU 上执行，但深度学习训练通常需要 GPU 加速。如果代码写死用 GPU，在没有 GPU 的机器上会报错；如果写死用 CPU，则浪费了 GPU 的性能。因此，工业标准做法是**自动检测**：有 GPU 就用 GPU，没有就用 CPU。

### 核心代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 自动选择设备：有 GPU 用 GPU，否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")
```

### 逐行拆解

1. **`import torch` 等导入语句**：引入 PyTorch 核心库、神经网络模块（`nn`）、优化器（`optim`）以及数据处理工具（`Dataset`, `DataLoader`）。这些是训练循环的"工具箱"。
2. **`torch.cuda.is_available()`**：检查当前环境是否支持 CUDA（即是否有可用的 NVIDIA GPU）。
3. **`torch.device(...)`**：根据检查结果，创建一个设备对象。如果支持 CUDA，设备为 `"cuda"`；否则为 `"cpu"`。
4. **`print(...)`**：打印当前使用的设备，方便你确认代码运行环境。

### 关键理解

- **`device` 是一个变量**，后续定义模型、数据时，都需要通过 `.to(device)` 将对象移动到该设备上（本块暂不涉及，后续模块会用到）。
- **这行代码是通用的**：无论你在个人电脑（CPU）还是服务器（GPU）上运行，都不需要修改代码。这是工程化训练脚本的标准开头。

### 常见误区

- 不要直接写 `device = "cuda"`，除非你确定运行环境一定有 GPU。否则在无 GPU 的机器上会直接报错。
- 不要忽略 `print` 语句。它能帮你快速定位问题：如果训练速度异常慢，先检查是否意外使用了 CPU。

### 操作验证

在 `pytorch` conda 环境中运行上述代码，观察输出：
- 如果显示 `当前使用设备: cuda`，说明你的环境有 GPU 且已正确配置。
- 如果显示 `当前使用设备: cpu`，说明当前环境

## 块 2

```python
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data_num):
        self.x = torch.randn(data_num, 2)
        # 简单分类规则：x0 + x1 > 0 则为类别 1，否则为 0
        self.y = torch.where(self.x[:, 0] + self.x[:, 1] > 0,
                             torch.tensor(1), torch.tensor(0))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

**关键点**：你只需要实现`__len__`（返回样本总数）和`__getitem__`（按索引返回一个样本及其标签）。PyTorch的`Dataset`基类会自动帮你处理剩下的。

**划分训练集和验证集**

数据不能全用来训练，要留一部分“考”模型。这里直接创建两个`SimpleDataset`实例，一个用于训练，一个用于验证：

```python
train_dataset = SimpleDataset(800)
val_dataset = SimpleDataset(200)
```

这样，训练集有800个样本，验证集有200个样本，两者独立生成，互不干扰。

## 块 3

## 3.3 模型、损失函数、优化器（①）——为LoRA微调打基础

你正在学习LoRA微调，但LoRA不是凭空出现的——它是在一个完整的训练循环里工作的。这一块我们先看清“训练循环”的全貌，特别是**模型、损失函数、优化器**这三个核心组件如何配合。

### 模型：SimpleModel

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),   # 输入2维 → 32维
            nn.ReLU(),
            nn.Linear(32, 64),  # 32维 → 64维
            nn.ReLU(),
            nn.Linear(64, 2)    # 64维 → 输出2维
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel().to(device)
```

这是一个三层全连接网络。输入是2个特征，输出是2个类别。`nn.Sequential`把层按顺序串起来，数据从输入到输出依次流过。

### 损失函数与优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

- **损失函数**：`CrossEntropyLoss`，用于分类任务，衡量预测和真实标签的差距。
- **优化器**：`AdamW`，负责根据梯度更新模型参数。`lr=1e-3`是学习率，`weight_decay=0.01`是权重衰减（防止过拟合）。

### 学习率调度器（可选但推荐）

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
```

训练过程中动态调整学习率，让训练更平稳。`T_max=30`表示30轮内从初始学习率余弦下降到`eta_min=1e-6`。

---

## 四、训练阶段：train_one_epoch（①+②）

把一轮训练封装成函数，是工程代码的标准写法。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()           # 启用训练模式（Dropout、BatchNorm 行为切换）
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # ===== 训练五步走 =====（②）
        # 1. 前向传播
        outputs = model(features)
        loss = criterion(outputs, labels)

        # 2. 清零梯度（必须！否则梯度会累加）
        optimizer.zero_grad()

        # 3. 反向

## 块 4

## 验证循环：评估模型表现

训练完模型后，你需要一个“考试环节”来检验它学得怎么样。下面这个函数就是干这件事的：

```python
def val_one_epoch(model, loader, criterion, device):
    model.eval()            # 启用评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    # 关闭梯度计算，节省显存、加速推理
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += features.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc
```

**两个必须**：验证阶段必须调用 `model.eval()`，必须用 `with torch.no_grad()` 包裹。两者缺一不可，否则不仅评估结果不准，还可能因构建计算图而浪费大量显存。

**动手试一试**：把这段代码跑一遍，观察 `model.eval()` 和 `torch.no_grad()` 分别

## 块 5

## 三、准备工作：模型、损失函数、优化器（①）——训练循环与模型保存

### 训练主循环：30轮迭代

下面这段代码是训练的核心骨架。你有一个目标：**用验证集准确率挑出最好的模型**，而不是盲目训练完就结束。

> **注意：训练 LoRA 模型需要至少 8GB 显存**，请确保你的环境满足要求后再运行以下代码。

```python
EPOCHS = 30
best_val_acc = 0.0

# 用于记录指标，后续绘图
train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []

print("========= 开始训练 ==========")
for epoch in range(EPOCHS):
    try:
        # 训练一轮 + 验证一轮
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)
    except Exception as e:
        print(f"训练或验证过程出错：{e}")
        break

    # 更新学习率（如果使用调度器）
    scheduler.step()

    # 记录指标
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    # 打印日志
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    print("-" * 50)

    # 保存最优模型（验证集准确率最高时）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        try:
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  >>> 发现更优模型！验证准确率 {best_val_acc:.4f}，已保存")
        except Exception as e:
            print(f"保存最优模型失败：{e}")

# 保存最后一轮模型
try:
    torch.save(model.state_dict(), "last_epoch_model.pth")
    print(f"\n训练结束！最高验证准确率：{best_val_acc:.4f}")
except Exception as e:
    print(f"保存最后一轮模型失败：{e}")
```

**逐段拆解，动手跟练：**

1. **初始化**：`best_val_acc = 0.0` 用来追踪目前见过的最好验证准确率。四个空列表分别存训练/验证的损失和准确率，后面画曲线用。

2. **循环体**：每一轮（epoch）做两件事——先训练一轮，再验证一轮。`train_one_epoch` 和 `val_one_epoch` 是封装好的函数，你只需要传入模型、数据加载器、损失函数、优化器和设备。代码已用 `try-except` 包裹，出错时会打印错误信息并退出循环。

3. **学习率调度**：`scheduler.step()` 每轮更新一次学习率。如果你没定义 scheduler，这行可以注释掉，不影响主流程。

4. **记录与打印**：把本轮指标追加到列表里，然后格式化打印。注意 `:.4f` 保留4位小数。

5. **保存最优模型**：核心逻辑在 `if val_acc > best_val_acc` 判断中——只有验证准确率刷新纪录时才保存，确保最终留下的是表现最好的模型

## 块 6

## 三、准备工作：模型、损失函数、优化器（①）——可视化与排错

### 3.3.1 用图表“看见”训练过程

训练循环跑起来后，光看终端里的数字不够直观。用 `matplotlib` 把损失和准确率画成曲线，能一眼看出模型是在收敛、过拟合还是根本没学进去。

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
try:
    plt.savefig("training_curve.png")
    print("训练曲线已保存至 training_curve.png")
except Exception as e:
    print(f"保存训练曲线失败：{e}")
```

**读图要点**：左图看损失下降趋势，右图看准确率上升趋势。两条曲线（训练/验证）的间距是关键信号——间距拉大往往意味着过拟合。

> **显存提示**：训练时需注意GPU显存占用，建议使用梯度累积或减小batch_size。

### 3.3.2 常见问题速查表

训练中报错或指标异常时，对照下表快速定位：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 损失震荡严重，无法收敛 | 忘记 `optimizer.zero_grad()` | 每次 `backward()` 前必须清零梯度 |
| 验证准确率远低于训练准确率 | 过拟合 | 增大 `weight_decay`、增加数据、添加 Dropout |
| 验证阶段结果不稳定 | 忘记 `model.eval()` | 验证前务必调用 `model.eval()` |
| GPU 显存爆炸 | 验证阶段未用 `torch.no_grad()` | 验证代码必须用 `with torch.no_grad()` 包裹 |
| `RuntimeError: Expected all tensors on the same device` | 模型和数据不在同一设备 | 确保 `model.to(device)` 和数据 `.to(device)` 都执行 |
| 模型加载报错 | 直接保存整个模型 `torch.save(model)` | 改用 `state_dict` 方式保存和加载 |
| 验证集指标波动大 | `shuffle=True` | 验证集 DataLoader 必须 `shuffle=False` |

**动手排查建议**：下次训练出问题时，先别急着改代码。对照上表，逐条检查你的训练循环——特别是 `zero_grad()` 和 `model.eval()` 这两个最容易被忽略的步骤。

### 3.3.3 本节小结

你现在应该掌握了：

- ✅ 用 `matplotlib` 绘制损失/准确率曲线，直观监控训练过程
- ✅ 常见训练问题的症状—原因—解决方案对照排查
- ✅ `model.train()` / `model.eval()`、`torch.no_grad()` 的正确使用场景

> **模型许可证**：本教程涉及的模型需遵守其许可证条款。例如，Llama 2 采用 Llama 2 Community License，商用月活用户超过 7 亿时需获得 Meta 授权。使用前请确认所选模型的许可证类型。

> **下一步预告**：下一节 **"模型保存与加载（checkpoint）"** 将深入讲解模型持久化：`state_dict` 之外，还有保存优化器状态（断点续训）、多卡模型保存、HuggingFace

## 块 7

## 方式一：保存整个模型（不推荐）

先看代码——你之前训练循环里那行 `torch.save(model.state_dict(), "best_model.pth")`，保存的是**参数**。而方式一是这样：

```python
torch.save(model, "model.pkl")
```

区别就在：`model` 和 `model.state_dict()`。前者把**整个模型对象**打包，包括结构定义、参数等；后者只存参数。

**为什么官方不推荐方式一？** 四个坑：

1. **文件体积大**——结构定义也占空间，存了很多你用不到的东西。
2. **加载时要求模型类定义可用且路径一致**——换个项目目录，类找不到，直接报错。
3. **跨 PyTorch 版本或跨项目容易报错**——版本升级后，旧格式可能不兼容。
4. **无法灵活地只加载部分参数**——比如你想只恢复某一层的权重，方式一做不到。

**对你的 LoRA 微调意味着什么？** 你后面做 LoRA 时，核心操作是**只加载预训练权重的一部分**（冻结原模型参数，只训练低秩矩阵）。方式一这种"全量打包"的保存方式，在 LoRA 场景下完全不可用——你根本不需要、也不想要整个模型的结构被序列化进去。你需要的是**精确控制哪些参数被保存、哪些被加载**，这正是方式二（`state_dict`）的强项。

**动手验证**：在 `pytorch` conda 环境里跑一下这两行，对比 `model.pkl` 和 `best

## 块 8

**本块学习检查**：读完本块后，请合上资料，用一句话回答：保存 `state_dict` 相比保存整个模型，最大的两个优势是什么？

---

**保存 state_dict：只存“参数”，不存“结构”**

`model.state_dict()` 返回一个**有序字典**，里面每个键是参数名（如 `fc1.weight`），值是张量。它只存**可学习参数和缓冲区**（如 BatchNorm 的 `running_mean`），**不包含模型结构定义**。

保存只需一行：
```python
torch.save(model.state_dict(), "model_state_dict.pkl")
```

**为什么官方推荐它？**

- **文件小**：只存参数，体积远小于整个模型
- **加载灵活**：先实例化一个相同结构的新模型，再加载即可
- **跨设备方便**：CPU/GPU 之间迁移无障碍
- **工业标准**：生产环境、部署、跨项目协作都靠它

**对比速

## 块 9

### 加载整个模型（方式一）

最直接的方式是把整个模型对象保存下来，加载时一步到位：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
try:
    model = torch.load("model.pkl", map_location='cpu')
except Exception as e:
    print(f"模型加载失败：{e}")
    raise
model.eval()
```

**注意**：加载后必须调用 `model.eval()` 切换到评估模式，否则模型会处于训练模式，影响推理结果。

---

### 加载 state_dict（方式二）

更推荐的方式是只加载参数，分三步走：

```python
# 1. 先实例化一个相同结构的模型
model = MyModel()

#

## 块 10

### 跨设备加载：`map_location` 实战

你正在做 LoRA 微调，很可能在 GPU 上训练、在 CPU 上做推理。这时直接 `torch.load()` 会报错——因为如果 checkpoint 是在 GPU 上保存的，而当前环境没有 GPU，直接 `torch.load()` 会报错，因为张量被加载到 CUDA 设备。

解决办法是加一个 `map_location` 参数，告诉 PyTorch“把模型加载到哪个设备上”。看代码：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 无论 checkpoint 是在 CPU 还是 GPU 上保存的，都加载到当前 device
checkpoint = torch.load("model_state_dict.pt", map_location=device)

model = MyModel()
model.load_state_dict(checkpoint)
model.to(device)
```

关键点：
- `map_location` 是**必备参数**，不是可选项。强烈建议每次加载都显式写出来，省得在不同设备间切换时报错。
- 代码里先定义 `device`，再传给 `map_location`，这样同一份代码在有无 GPU 的环境下都能跑。
- # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
- # 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 11

你的目标是 LoRA 微调，且偏好动手实践。这里我们聚焦一个关键动作：**把训练现场完整“存档”**，以便随时恢复。

**核心思路**：保存的不只是模型权重，而是能让你“无缝续跑”的完整状态包。

**动手写代码**：在训练循环中，定期打包以下内容：

```python
checkpoint = {
    "epoch": epoch,                           # 当前训练轮次
    "model_state_dict": model.state_dict(),   # 模型参数
    "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,  # 学习率调度器状态
    "best_val_acc": best_val_acc,             # 当前最优指标
}

torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
```

**保存时机建议**：
- 每 5~10 个 epoch 存一次常规存档
- 验证集指标创新高时，额外存一份 `checkpoint_best.pth`
-

## 块 12

### 恢复 checkpoint：让训练从断点继续

你已经跑完了10个epoch，模型学到了不少东西。现在要做的不是从头再来，而是把之前保存的“快照”完整恢复，接着往下训。整个过程分6步，每一步对应保存时的一个部分。

**第1步：加载文件**
```python
checkpoint = torch.load("checkpoint_epoch_10.pth", map_location=device)
```
`map_location=device` 确保模型加载到当前设备（GPU或CPU），避免设备不匹配报错。  
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。  
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**第2步：恢复模型参数**
```python
model = MyModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
```
先新建一个模型实例，再把checkpoint里保存的权重灌进去。注意：模型结构必须和保存时一致，否则`load_state_dict`会报错。

**第3步：恢复优化器状态**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```
优化器也要新建，然后恢复它的状态（比如动量、学习率调整历史）。这样后续训练的学习率节奏才能衔接上。

**第4步：恢复学习率调度器**
```python
if checkpoint["scheduler_state_dict"] is not None:
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
```
调度器可能没保存（为`None`），所以先判断再恢复。

**第5步：恢复训练进度**
```python
start_epoch = checkpoint["epoch"] + 1
```
从保存的epoch的下一个epoch开始继续训练，避免重复训练已完成的轮次。

**第6步：继续训练循环**
```python
for epoch in

## 块 13

## 断点续训完整示例：让训练“断电不慌”

你已经掌握了 checkpoint 的基本概念，现在我们把它们串成一个完整的实战脚本。这个示例就是 LoRA 微调时最常用的训练骨架——**先尝试恢复，再开始训练，定期存档**。

### 整体流程拆解

整个脚本分三步走：

1. **初始化**：定义模型、优化器、损失函数，并设置训练参数。
2. **尝试恢复**：加载已有的 checkpoint 文件，把模型和优化器的状态“接回去”。
3. **训练循环**：从断点处继续训练，并每隔固定轮数保存一次。

### 关键代码逐段看

**第一步：准备阶段**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
CHECKPOINT_INTERVAL = 5
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```

- `device`：自动选择 GPU 或 CPU，你不需要手动改。
- `EPOCHS = 20`：总共训练 20 轮。
- `CHECKPOINT_INTERVAL = 5`：每 5 轮存一次档。

**第二步：恢复 checkpoint（核心中的核心）**

```python
resume_path = "checkpoint_latest.pth"
try:
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    print(f"成功恢复训练，从 epoch {start_epoch} 继续")
except FileNotFoundError:
    print("未找到 checkpoint，从头开始训练")
```

这里用了 `try-except` 结构，逻辑很清晰：

- **如果文件存在**：加载模型参数、优化器状态，然后从 `epoch + 1` 继续（因为 `epoch` 是已经完成的轮数，下一轮要加 1）。
- **如果文件不存在**：捕获 `FileNotFoundError`，从头开始，`start_epoch` 保持为 0。

> **动手验证**：你可以先不创建 checkpoint 文件运行一次，观察输出；再手动创建一个空的 `checkpoint_latest.pth` 文件（内容随意），再次运行，对比两次输出的差异。

**第三步：训练循环与定期保存**

```python
for epoch in range(start_epoch, EPOCHS):
    model.train()
    # ... 训练步骤 ...

    if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS - 1:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(checkpoint, "checkpoint_latest.pth")
        print(f"Epoch {epoch+1}: checkpoint 已保存")
```

保存时打包了四个关键信息：

- `epoch`：当前已完成轮数，用于恢复时计算下一轮起点。
- `model_state_dict`：模型权重，恢复后模型参数与保存时完全一致。
- `optimizer_state_dict`：优化器状态（如动量、学习率调度位置），确保恢复后训练节奏不中断。
- `best_acc`：历史最佳

## 块 14

### 5.1 内存映射加载：mmap=True（③）

你正在做LoRA微调，模型文件往往很大。传统 `torch.load` 会把整个文件一次性读入内存，RAM 不够就容易卡死。改用 `mmap=True`，让操作系统**按需加载**数据块——用到哪块读哪块，避免一次性占用大量内存：

```python
state_dict = torch.load("large_model.pth", mmap=True, map_location="cpu")
```

### 5.2 元设备初始化：torch.device('meta')（③）

加载大模型前，先用**元设备**创建模型——它只记录张量的形状和属性，不分配实际内存。等真正加载参数时再按需填充，适合微调前快速搭建骨架：

```python
# 1. 在元设备上创建模型（零内存占用）
with torch.device('meta'):
    model = LargeModel()

# 2. 加载 state_dict（同样使用 mmap）
state_dict = torch.load("large_model.pth", mmap=True)

# 3. 赋值式

## 块 15

### 5.3 三种技巧速查（③）

你正在为LoRA微调打基础，这里三个技巧是后续高频使用的“加速器”，按场景记即可：

| 技巧 | 作用 | 适用场景 |
|---|---|---|
| `mmap=True` | 将文件映射到虚拟内存，加快加载速度，但不减少内存峰值 | 任何大模型加载 |
| `torch.device('meta')` | 创建模型时不分配实际内存，仅记录元数据，节省内存 | 加载后参数会被覆盖的场景 |
| `assign=True` | 直接将加载的张量赋值给模型参数（in-place），避免额外拷贝 | 在需要避免参数拷贝时使用 |

**实操理解**：`mmap=True` 让模型文件映射到虚拟内存，适合你日常2.5小时训练中反复加载大模型时加快加载速度。`meta` 设备则是在你确定马上要用 checkpoint 覆盖参数时，省去先分配实际内存再覆盖的浪费。`assign=True` 是最后一步，把加载的权重直接“贴”到模型上。

---

### 六、常见问题与解决（①+②）

训练中遇到报错，先对照这张表：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `RuntimeError: Error(s) in loading state_dict` | 模型结构不一致 | 确保加载前实例化的模型结构与保存时完全一致 |
| GPU 训练保存，CPU 加载报错 | 设备不匹配 | 使用 `map_location=device` |
| 恢复训练后损失突然跳变 | 忘记恢复优化器状态 | checkpoint 必须包含 `optimizer_state_dict` |
| 恢复训练后学习率不对 | 忘记恢复 scheduler | checkpoint 包含 `scheduler_state_dict` |
| 同一 epoch 被重复训练 | `start_epoch` 没 +1 | 恢复后设置 `start_epoch = checkpoint["epoch"] + 1` |
| checkpoint 文件过大 | 保存了整个模型对象 | 改用 `model.state_dict()` 方式保存 |
| 加载速度极慢 | 大模型 + 传统加载 | 使用 `mmap=True` 或优化存储格式（如safetensors） |

**重点提醒**：你正在学LoRA，后续微调时最常踩的坑是“恢复训练后损失跳变”——这几乎都是因为只恢复了

## 测评题目

### Q-KB0406-001
训练循环中，`optimizer.zero_grad()` 的作用是？

### Q-KB0406-003
请补全以下训练循环代码：
```python
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        # 前向传播
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        # 反向传播
        ________
        # 更新参数
        ________
```

### Q-KB0407-004
加载checkpoint时，`map_location` 参数的作用是？

### Q-KB0407-001
保存模型**参数字典**的正确方法是？

### Q-KB0401-001
以下哪个命令可以创建一个全为0的3x3张量？

### Q-KB0402-003
以下代码输出什么？
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 3
z.backward()
print(x.grad)
```

### Q-KB0401-003
请将以下两个张量按**列方向**（dim=1）拼接：
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5], [6]])
```

### Q-KB0404-001
自定义数据集类需要继承的基类是？
