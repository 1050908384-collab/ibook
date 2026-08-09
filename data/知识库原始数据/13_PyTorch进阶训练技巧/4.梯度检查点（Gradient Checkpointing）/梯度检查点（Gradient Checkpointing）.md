# 梯度检查点（Gradient Checkpointing）

① 底稿：《梯度检查点技术的显存与计算权衡：在训练时长和显存之间找最优解》  
　　网址：https://blog.csdn.net/lady_mumu/article/details/162865301

② 参考文章：《torch.utils.checkpoint模块介绍》  
　　网址：https://blog.csdn.net/qq_27390023/article/details/144833218

③ 参考文章：《梯度检查点技术实战指南》  
　　网址：https://blog.csdn.net/gamma/article/details/155682214

> **适合人群**：已完成"混合精度训练（AMP）"章节，理解训练循环、反向传播、显存占用构成的初学者。本节时长约 1 小时，目标是理解梯度检查点"以计算换显存"的核心原理，掌握 PyTorch 的 `torch.utils.checkpoint` 使用方法，能够根据显存瓶颈选择合适的 checkpoint 粒度，并与 AMP、梯度累积配合使用。（①）
>
> **前置说明**：上节我们学习了 AMP，它通过 FP16 计算在特定条件下可将显存降低约 30%~50%（取决于模型结构、GPU 架构和 batch size）。但对于深层 Transformer（如 32 层的 LLaMA-7B），**激活值（Activations）** 的显存占用可能与模型参数本身相当——每层约 256MB，32 层就是 8GB。当 batch size 稍大或序列较长时，激活值会成为压垮 GPU 的主要瓶颈之一。**梯度检查点（Gradient Checkpointing）** 是解决这个问题的重要手段：它丢弃大部分中间激活值，只在反向传播需要时重新计算。代价是什么？在特定条件下约 20%~40% 的额外训练时间。本节将带你理解这个权衡，并做出合适的选择。

## 一、为什么需要梯度检查点？

### 1.1 激活值：隐形的显存杀手

在训练神经网络时，显存被三部分占用：

1. **模型参数**：固定，与参数量成正比
2. **优化器状态**：固定，Adam 约为参数量的 2 倍
3. **激活值（Activations）**：**与层数、batch size、序列长度成正比**

对于 Transformer，每层前向传播产生的激活值需要保留到反向传播使用。一个 $L$ 层模型的激活值显存约为：（①）

$$\text{Activation Memory} \approx B \times L \times H \times S \times \text{bytes\_per\_element}$$

其中：
- $B$：batch size
- $L$：层数
- $H$：隐藏维度
- $S$：序列长度

**实例：LLaMA-7B**（①）

- $L = 32$，$H = 4096$，$S = 2048$，$B = 4$，FP16（2 bytes）
- 每层激活值：$4 \times 4096 \times 2048 \times 2 \approx 64\ \text{MB}$（注：实际因多头注意力和 FFN 会更大，底稿给出约 256MB/层）
- 32 层总激活值：**约 8 GB**

这已经与模型参数（FP16 约 14 GB）相当。加上优化器状态，单卡 40GB 的 A100 很快捉襟见肘。

### 1.2 核心思想：以计算换显存

标准训练流程：
- **前向**：计算每一层，保留所有中间激活值 → 显存 $O(L)$
- **反向**：从最后一层开始，用保存的激活值计算梯度

梯度检查点流程：
- **前向**：计算每一层，但**只保留少数"检查点"层的激活值**，丢弃其余 → 显存 $O(1)$ 或 $O(\sqrt{L})$
- **反向**：当需要某段中间的激活值时，**用保存的检查点重新执行该段前向传播**，得到激活值后立即计算梯度并释放

> **关键洞察**：我们牺牲了部分前向计算时间（重算），换取了显存的显著降低。对于计算资源充足但显存紧张的场景，在特定条件下这是一种可接受的权衡。（①）

## 二、PyTorch 实现：从简单到复杂

PyTorch 提供了 `torch.utils.checkpoint` 模块，无需手动实现复杂的重计算逻辑。

### 2.1 基础 API：`checkpoint` 函数

```python
from torch.utils.checkpoint import checkpoint

# 假设 layer 是一个 nn.Module，x 是输入
output = checkpoint(layer, x, use_reentrant=False)
```

**参数说明**：（②）

| 参数 | 说明 |
|------|------|
| `function` | 需要检查点的模块或函数 |
| `*args` | 输入参数 |
| `use_reentrant` | `True`：旧版实现，反向时重新调用 `forward`；`False`：使用 saved tensor hooks，推荐 |
| `preserve_rng_state` | 是否保留随机状态（确保 dropout 等随机操作重计算时一致） |

### 2.2 在 Transformer 中使用

以下是一个完整的、集成梯度检查点的 Transformer 实现：（①）

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class TransformerWithCheckpointing(nn.Module):
    def __init__(self, n_layers=12, d_model=768, checkpoint_every=1, use_reentrant=False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=12,
                dim_feedforward=4 * d_model,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.checkpoint_every = checkpoint_every
        self.use_reentrant = use_reentrant

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 只在训练时启用 checkpoint，推理时无额外开销
            # 注意：这是示例模型自己实现的条件判断，非 PyTorch API 的自动行为
            if self.training and i % self.checkpoint_every == 0:
                x = checkpoint(
                    layer, x,
                    use_reentrant=self.use_reentrant,
                    preserve_rng_state=True,
                )
            else:
                x = layer(x)
        return x
```

> **关键细节**：`checkpoint` 本身不会自动根据 `train()`/`eval()` 模式决定是否启用，上述 `if self.training` 是**示例模型自己实现的条件判断**，非 PyTorch API 的自动行为。实际使用时需自行控制，以确保推理时不产生额外开销。（①）

### 2.3 `use_reentrant` 参数：选 True 还是 False？

这是 PyTorch 1.11+ 引入的重要参数，直接影响正确性和性能：（①）

| 特性 | `use_reentrant=True`（旧版） | `use_reentrant=False`（推荐） |
|------|---------------------------|---------------------------|
| **实现机制** | 反向时重新调用 `forward()` | 使用 autograd 的 saved tensor hooks |
| **重入要求** | 要求 `forward()` 可重入（两次调用结果相同） | 无重入要求 |
| **与数据并行的兼容性** | 可能有问题 | 通常与 DDP 兼容 |
| **性能** | 需要重建计算图，开销较大 | 通常更高效，计算图更轻量 |
| **推荐版本** | PyTorch < 1.11 | PyTorch ≥ 1.11 |

> **建议**：如果你使用 PyTorch 1.11 或更高版本，**通常建议设置 `use_reentrant=False`**。它在特定条件下不仅更快，还能避免许多与随机性、状态更新相关的 bug。（①）

### 2.4 快捷方式：`checkpoint_sequential`

如果你的模型是一连串顺序执行的层，可以使用更简洁的 API：（②）

```python
from torch.utils.checkpoint import checkpoint_sequential

# 将 12 层分成 3 段，每段 4 层
output = checkpoint_sequential(
    self.layers,      # nn.Sequential 或 nn.ModuleList
    segments=3,       # 分成几段
    input=x,
    use_reentrant=False,
)
```

`segments` 参数决定了检查点的粒度：
- `segments=1`：整个序列作为一个检查点（显存最小，重算最多）
- `segments=L`：每层都是检查点（显存较大，重算最少）

## 三、粒度选择：在显存和时间之间找平衡

### 3.1 实测数据：LLaMA-7B 上的对比

底稿①在 LLaMA-7B（配置：$L=32, H=4096$，FP16，batch_size=4，seq_len=2048，NVIDIA A100 40GB）上测试了不同 `checkpoint_every` 配置的效果：

| 策略 | 峰值显存 | 训练时间 | 相对计算量 |
|------|---------|---------|-----------|
| 无 checkpoint | 68 GB | 100% | 1.0× |
| `checkpoint_every=4` | 52 GB | 112% | 1.12× |
| `checkpoint_every=2` | 38 GB | 122% | 1.22× |
| `checkpoint_every=1` | 28 GB | 138% | 1.38× |

**关键观察**：（①）

1. **显存节省显著**：`checkpoint_every=1` 时显存从 68GB 降至 28GB，在该测试条件下节省了约 **60%**
2. **计算开销低于直觉**：每层都 checkpoint 的计算开销在该测试条件下约 38%，而不是 naive 预期的 100%。原因是：
   - 重计算只发生在反向传播阶段
   - `use_reentrant=False` 通过 saved tensor hooks 优化了计算图重建
3. **边际递减**：从 `every=4` 到 `every=2` 节省 14GB，代价 10% 时间；从 `every=2` 到 `every=1` 节省 10GB，代价 16% 时间

> **注意**：上述数据来自底稿①在特定配置下的测试，实际数值会因模型结构、硬件、PyTorch 版本等因素而异。

### 3.2 如何选择最优粒度？

根据你的显存缺口和训练瓶颈，按以下决策树选择：（①）

```
瓶颈是什么？
├── 显存不足（OOM）
│   ├── 缺口 < 20% → checkpoint_every=4（轻量节省）
│   ├── 缺口 20%~40% → checkpoint_every=2（中等节省）
│   └── 缺口 > 40% → checkpoint_every=1（最大节省）
│       └── 仍不足？→ 结合 CPU Offload 或模型并行
├── 计算时间太长
│   └── 减少 checkpoint 频率，或仅在深层使用
└── 想增大 batch size
    └── 优先增大 batch + checkpoint，总吞吐可能更高
```

> **工程经验**：不建议一上来就 `checkpoint_every=1`。建议从 `every=2` 或 `every=4` 开始，观察显存占用，逐步收紧直到刚好不爆显存。这样能在显存安全和训练速度之间找到较合适的平衡点。（①）

## 四、与 AMP + 梯度累积的配合

梯度检查点、AMP、梯度累积是**显存优化的常见组合**，它们可以叠加使用，效果互补：（①）

| 技术 | 节省什么 | 代价 | 叠加效果 |
|------|---------|------|---------|
| **梯度累积** | 用时间换等效 batch size | 更新频率降低 | 与 checkpoint 无冲突 |
| **AMP** | 激活值和参数精度减半 | 可能的数值不稳定 | 降低 checkpoint 重计算时的显存压力 |
| **梯度检查点** | 激活值显存 | 20%~40% 额外计算（特定条件下） | 与 AMP 配合时，重计算也用 FP16，更快更省 |

**三合一的标准训练模板**：

```python
from torch import amp
from torch.utils.checkpoint import checkpoint

scaler = amp.GradScaler()
accumulation_steps = 4
checkpoint_every = 2

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        with amp.autocast(device_type='cuda'):
            # 在模型内部，每层根据 checkpoint_every 决定是否使用 checkpoint
            outputs = model(inputs)  # model 内部已集成 checkpoint
            loss = criterion(outputs, labels) / accumulation_steps
        
        scaler.scale(loss).backward()  # 反向时自动触发重计算
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

> **注意**：`checkpoint` 与 `autocast` 通常兼容。重计算的前向传播也会在 `autocast` 的上下文中自动使用 FP16，因此通常不会显著增加额外的显存负担。（①）

## 五、进阶：手动实现自定义 Checkpoint

在某些特殊场景下（如 forward 中有不可重入的操作、需要精细控制哪些张量保存），你可以手动实现 checkpoint：（①）

```python
import torch
from torch.autograd import Function

class CustomCheckpointFunction(Function):
    """手动实现的自定义 checkpoint
    
    使用场景：
    - forward 中有状态更新（如全局计数器），不可重入
    - 需要选择性保留某些中间结果，避免不必要重算
    """
    
    @staticmethod
    def forward(ctx, x, weight, bias):
        # 只保存反向所需的最小信息
        ctx.save_for_backward(x, weight, bias)
        
        # 前向计算，不保留中间结果
        with torch.no_grad():
            y = torch.matmul(x, weight.t()) + bias
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        
        # 反向时重新计算前向（重计算）
        y = torch.matmul(x, weight.t()) + bias
        
        # 基于重计算的 y 计算梯度
        grad_x = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), x)
        grad_bias = grad_output.sum(dim=0)
        
        return grad_x, grad_weight, grad_bias

# 使用方式
custom_linear = CustomCheckpointFunction.apply
output = custom_linear(x, weight, bias)
```

> **警告**：手动实现在特定条件下容易出错，除非有特殊需求，否则优先使用 PyTorch 官方的 `torch.utils.checkpoint.checkpoint`。（③）

## 六、常见问题与解决（①+③）

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| "开启 checkpoint 后训练变慢 50% 以上" | `use_reentrant=True` 或 checkpoint 粒度过细 | 升级到 PyTorch 1.11+，设置 `use_reentrant=False`；增大 `checkpoint_every`（①） |
| "checkpoint 在推理时也触发重计算" | 未区分 train/eval 模式 | `checkpoint` 本身不会自动根据模式切换，需在代码中自行控制（如示例中的 `if self.training`），确保推理时不产生额外开销（①） |
| "随机操作（dropout）结果不一致" | 重计算时的随机状态与首次前向不同 | 设置 `preserve_rng_state=True`（默认），或固定随机种子（③） |
| "与 DDP 分布式训练冲突" | `use_reentrant=True` 与 DDP 的梯度同步可能不兼容 | 使用 `use_reentrant=False`（①） |
| "某些层不能 checkpoint（如 BatchNorm）" | BatchNorm 的 running statistics 在重计算时更新 | 对 BatchNorm 层跳过 checkpoint，或改用 LayerNorm（③） |
| "显存节省不如预期" | 模型中有大量不可 checkpoint 的操作 | 检查是否所有层都包裹了 checkpoint；某些自定义操作可能不支持（①） |

## 七、总结与后续学习路线（①）

完成本节学习后，你应该掌握了：

- ✅ **为什么需要梯度检查点**：深层 Transformer 的激活值显存占用巨大，是 OOM 的常见原因之一
- ✅ **核心原理**：丢弃中间激活值，反向传播时按需重计算，以 20%~40% 的额外计算时间（特定条件下）换取 30%~60% 的显存节省（特定条件下）
- ✅ **PyTorch 实现**：`torch.utils.checkpoint.checkpoint` 和 `checkpoint_sequential` 的使用
- ✅ **`use_reentrant=False`**：PyTorch 1.11+ 的通常建议设置，更高效且通常与 DDP 兼容
- ✅ **粒度选择**：根据显存缺口选择 `checkpoint_every`（1/2/4），在显存安全和速度之间找平衡
- ✅ **三合一配合**：梯度检查点 + AMP + 梯度累积是显存优化的常见组合

> **下一步预告**：在下一节 **"学习率调度（Warmup + 余弦退火）"** 中，我们将跳出显存优化的视角，从训练动态的角度学习如何让模型收敛得更快、更稳——为什么大模型训练初期需要 Warmup？余弦退火相比固定学习率有什么优势？以及学习率调度与 batch size、优化器选择之间的联动关系。现在，试着修改上面的代码，将 `checkpoint_every` 分别设为 1、2、4，在相同模型和数据上运行，实测并对比峰值显存占用和训练时间！