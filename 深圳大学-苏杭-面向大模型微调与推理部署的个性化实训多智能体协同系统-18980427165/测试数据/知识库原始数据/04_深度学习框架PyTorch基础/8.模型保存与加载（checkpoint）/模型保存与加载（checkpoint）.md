# 模型保存与加载（checkpoint）

① 底稿：《【PyTorch框架】模型保存、加载与断点续训练》\
　　网址：https://blog.csdn.net/weixin_47244593/article/details/130134054

② 参考文章：《PyTorch Checkpoint保存与加载：断点续训关键步骤》\
　　网址：https://blog.csdn.net/weixin_32445049/article/details/156397856

③ 参考文章：《PyTorch教程：高效加载模型检查点的三大技巧》\
　　网址：https://blog.csdn.net/gitblog_01144/article/details/148440386

> **适合人群**：已完成"训练循环编写"章节，能够编写包含训练、验证和指标监控的完整 PyTorch 训练脚本的初学者。本节时长约 1 小时，目标是掌握 PyTorch 模型保存与加载的完整机制，能够正确使用 `state_dict` 保存和恢复模型，实现断点续训，并了解大模型场景下的高效加载技巧。（①+②）
>
> **前置说明**：上节我们在训练循环中用 `torch.save(model.state_dict(), "best_model.pth")` 保存了验证集表现最好的模型。本节将深入讲解这行代码背后的原理，以及更完整的 checkpoint 机制——包括如何同时保存优化器状态以实现"断点续训"。所有代码请在已安装 PyTorch 的 `pytorch` conda 环境中运行。

## 一、为什么需要保存模型？（①）

训练一个神经网络可能需要数小时甚至数天。模型在内存中以 Python 对象的形式存在，一旦程序结束或断电，这些对象就会被销毁。保存模型的本质是将内存中的对象**序列化**（转换成二进制数据）写入硬盘，实现长期存储。（①）

> **官方文档**：PyTorch 保存和加载模型完整指南 https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

## 二、两种保存方式：保存整个模型 vs 保存 state_dict（①）

PyTorch 提供了两种保存模型的方式，官方**强烈推荐第二种**。

### 2.1 方式一：保存整个模型（不推荐）（①）

```python
torch.save(model, "model.pkl")
```

这种方式把模型的**结构定义 + 参数 + 注册钩子**等全部打包保存。优点是简单，缺点是：（①）

-   文件体积大
-   加载时必须保证模型类定义可用（且路径一致）
-   跨 PyTorch 版本或跨项目时容易报错
-   无法灵活地只加载部分参数

### 2.2 方式二：保存 state_dict（官方推荐）（①）

```python
torch.save(model.state_dict(), "model_state_dict.pkl")
```

`state_dict()` 返回一个**有序字典**（OrderedDict），键是参数名（如 `fc1.weight`、`fc1.bias`），值是对应的张量。它只包含**可学习的参数和缓冲区**（如 BatchNorm 的 running_mean），不包含模型结构定义。（①）

**优点**：（①）

-   文件体积小
-   加载灵活，只需先实例化相同结构的模型
-   跨设备（CPU/GPU）加载方便
-   是工业标准做法

### 2.3 对比速查（①）

| 特性 | 保存整个模型 | 保存 state_dict |
|---|---|---|
| 保存内容 | 结构 + 参数 + 元数据 | 仅参数和缓冲区 |
| 文件大小 | 大 | 小 |
| 加载依赖 | 需要原始类定义 | 只需相同结构的新实例 |
| 官方推荐 | ❌ | ✅ |
| 适用场景 | 快速原型、同一文件内 | 生产环境、跨项目、部署 |

## 三、模型加载（①）

### 3.1 加载整个模型（对应方式一）（①）

```python
model = torch.load("model.pkl")
model.eval()
```

### 3.2 加载 state_dict（对应方式二）（①）

```python
# 1. 先实例化一个相同结构的模型
model = MyModel()

# 2. 加载参数
model.load_state_dict(torch.load("model_state_dict.pkl"))

# 3. 切换到评估模式
model.eval()
```

> **关键**：`load_state_dict()` 要求保存时的模型结构与新实例的结构**完全一致**。如果层名或形状不匹配，会报错 `RuntimeError: Error(s) in loading state_dict`。（①）

### 3.3 跨设备加载：map_location（①+②）

当模型在 GPU 上训练，需要在 CPU 上推理时，直接 `torch.load()` 会报错。使用 `map_location` 参数指定加载目标设备：（①+②）

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 无论 checkpoint 是在 CPU 还是 GPU 上保存的，都加载到当前 device
checkpoint = torch.load("model_state_dict.pkl", map_location=device)

model = MyModel()
model.load_state_dict(checkpoint)
model.to(device)
```

> `map_location` 是跨设备加载的**必备参数**，强烈建议始终显式指定。（②）

## 四、断点续训：保存与恢复完整训练状态（①+②）

上节的训练循环中，我们只保存了模型参数。但如果训练意外中断（断电、程序崩溃、服务器被抢占），重新启动时不仅要恢复模型参数，还需要恢复**优化器状态**（如 Adam 的动量缓存）和**训练进度**（当前 epoch），否则优化器会从"冷启动"状态重新开始，导致收敛异常。（①+②）

### 4.1 保存 checkpoint（①+②）

在训练循环中，定期保存一个包含完整状态的字典：

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

> **保存时机建议**：（②）
>
> -   每 5~10 个 epoch 保存一次常规 checkpoint
> -   验证集指标创新高时，额外保存 `checkpoint_best.pth`
> -   始终保留一份 `checkpoint_latest.pth`（软链接或覆盖）

### 4.2 恢复 checkpoint（①+②）

```python
# 1. 加载 checkpoint 文件
checkpoint = torch.load("checkpoint_epoch_10.pth", map_location=device)

# 2. 恢复模型参数
model = MyModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# 3. 恢复优化器状态
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# 4. 恢复学习率调度器
if checkpoint["scheduler_state_dict"] is not None:
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

# 5. 恢复训练进度
start_epoch = checkpoint["epoch"] + 1  # +1 避免重复训练同一轮
best_val_acc = checkpoint.get("best_val_acc", 0.0)

# 6. 继续训练
for epoch in range(start_epoch, total_epochs):
    # ... 训练循环 ...
```

> **注意 `epoch + 1`**：如果不加 1，会从已完成的 epoch 重新开始，导致该 epoch 被重复训练。（②）

### 4.3 断点续训完整示例（①）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# ========== 配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
CHECKPOINT_INTERVAL = 5

model = SimpleModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0.0

# ========== 尝试恢复 checkpoint ==========
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

# ========== 训练循环 ==========
for epoch in range(start_epoch, EPOCHS):
    model.train()
    # ... 训练步骤 ...

    # 定期保存 checkpoint
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

## 五、大模型高效加载技巧（拓展）（③）

当模型参数量达到数十亿甚至上百亿时（如 Llama、Qwen 等），传统的加载方式会遇到内存瓶颈。以下三种技巧可以显著降低加载时的内存占用。（③）

### 5.1 内存映射加载：mmap=True（③）

传统 `torch.load` 会将整个文件读入内存。使用 `mmap=True` 可以让操作系统**按需加载**数据块，避免一次性占用大量 RAM：

```python
state_dict = torch.load("large_model.pth", mmap=True, map_location="cpu")
```

### 5.2 元设备初始化：torch.device('meta')（③）

在加载大模型前，先用**元设备**创建模型——它只记录张量的形状和属性，不分配实际内存。加载参数时再按需填充：

```python
# 1. 在元设备上创建模型（零内存占用）
with torch.device('meta'):
    model = LargeModel()

# 2. 加载 state_dict（同样使用 mmap）
state_dict = torch.load("large_model.pth", mmap=True)

# 3. 赋值式加载（直接替换引用，避免复制）
model.load_state_dict(state_dict, assign=True)
```

### 5.3 三种技巧速查（③）

| 技巧 | 作用 | 适用场景 |
|---|---|---|
| `mmap=True` | 延迟加载，减少内存峰值 | 任何大模型加载 |
| `torch.device('meta')` | 跳过冗余初始化计算 | 加载后参数会被覆盖的场景 |
| `assign=True` | 直接替换参数引用 | 配合 meta device 使用 |

> 这三种技巧在后续"参数高效微调（LoRA）"和"模型推理部署（vLLM）"章节中会频繁用到。（③）

## 六、常见问题与解决（①+②）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `RuntimeError: Error(s) in loading state_dict` | 模型结构不一致 | 确保加载前实例化的模型结构与保存时完全一致 |
| GPU 训练保存，CPU 加载报错 | 设备不匹配 | 使用 `map_location=device` |
| 恢复训练后损失突然跳变 | 忘记恢复优化器状态 | checkpoint 必须包含 `optimizer_state_dict` |
| 恢复训练后学习率不对 | 忘记恢复 scheduler | checkpoint 包含 `scheduler_state_dict` |
| 同一 epoch 被重复训练 | `start_epoch` 没 +1 | 恢复后设置 `start_epoch = checkpoint["epoch"] + 1` |
| checkpoint 文件过大 | 保存了整个模型对象 | 改用 `model.state_dict()` 方式保存 |
| 加载速度极慢 | 大模型 + 传统加载 | 使用 `mmap=True` 和 `weights_only=True` |

## 七、总结与后续学习路线（①+②+③）

恭喜你完成了"深度学习框架PyTorch基础"板块的最后一节！至此，你应该掌握了：

-   ✅ 两种保存方式：保存整个模型 vs 保存 `state_dict`（推荐后者）
-   ✅ `state_dict` 的本质：有序字典，键为参数名，值为张量
-   ✅ 模型加载：`load_state_dict()` + `map_location` 跨设备加载
-   ✅ 断点续训：同时保存模型参数、优化器状态、epoch 和最优指标
-   ✅ checkpoint 恢复顺序：加载文件 → 恢复模型 → 恢复优化器 → 恢复 scheduler → 设置 start_epoch
-   ✅ 大模型高效加载：`mmap=True`、`meta` 设备、`assign=True`

> **下一步预告**：从下一节开始，我们进入 **"自然语言处理（NLP）基础概念"** 板块。你将学习分词、词性标注、命名实体识别等基础 NLP 任务，以及文本的数值化表示方法（Bag-of-Words、TF-IDF、Word2Vec）。随后我们将进入 **"注意力机制"** 和 **"Transformer 架构"** 章节，最终使用 PyTorch 加载预训练模型（如 GPT-2）进行前向传播实验——届时本节学到的 `torch.load` 和 `load_state_dict` 将成为你加载 HuggingFace 预训练权重的核心工具。现在，试着为你上节的训练脚本添加断点续训功能吧！