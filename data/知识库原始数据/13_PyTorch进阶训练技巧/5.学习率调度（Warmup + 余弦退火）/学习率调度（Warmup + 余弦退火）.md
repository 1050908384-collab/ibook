# 学习率调度（Warmup + 余弦退火）

① 底稿：《一文彻底搞懂：学习率预热Warmup与余弦退火，从原理、数学公式到SequentialLR工程实践》  
　　网址：https://blog.csdn.net/qq_62634342/article/details/161263879

② 参考文章：《从'余弦曲线'到'训练重启'：一文搞懂PyTorch中CosineAnnealingLR与WarmRestarts的异同与选型》  
　　网址：https://blog.csdn.net/weixin_42639633/article/details/161247769

③ 参考文章：《pytorch版本的学习率调度器》  
　　网址：https://blog.csdn.net/m0_48241022/article/details/144393152

④ 参考文章：《学习率调整策略，学习率预热（warmup）和学习率衰减（Learning Rate Decay）》  
　　网址：https://blog.csdn.net/weixin_45277161/article/details/131568962

> **适合人群**：已完成"梯度检查点"章节，理解训练循环、优化器、反向传播的初学者。本节时长约 1 小时，目标是理解学习率调度的必要性、掌握 Warmup + 余弦退火的数学原理与 PyTorch 工程实现，能够独立配置 `SequentialLR` 并避开常见陷阱。（①）
>
> **前置说明**：前面我们解决了训练的"硬件瓶颈"问题——通过 AMP 降低显存占用、通过梯度累积模拟大 batch、通过梯度检查点进一步压缩激活值。但即使显存和计算都够，如果学习率设置不当，模型依然可能训练失败。想象一下：你有一辆性能完美的赛车（优化好的训练流程），但如果没有一个聪明的变速箱（学习率调度），要么起步就爆缸（学习率过大导致梯度爆炸），要么全程龟速（学习率过小收敛极慢）。**Warmup + 余弦退火** 是现代大模型训练中较为常用、效果较好的学习率调度策略之一。本节将带你从原理到代码，掌握这套组合拳。

## 一、为什么需要学习率调度？

### 1.1 固定学习率的困境

在 PyTorch 基础板块中，我们使用了固定学习率（如 `lr=1e-4`）完成了训练。这在小型数据集和浅层网络上可行，但在深层 Transformer 上会遇到严重问题：（①）

**问题一：训练初期的梯度爆炸**

模型参数随机初始化时，前向传播可能产生极大的输出值，反向传播的梯度也相应巨大。如果此时学习率较大：

$$\Delta\theta = \eta \cdot \nabla L$$

参数更新量会成为"天文数字"，模型直接发散（Loss 变 NaN）。（①）

**问题二：Adam 优化器的动量偏差**

Adam/AdamW 依赖一阶矩 $m_t$ 和二阶矩 $v_t$ 的指数加权平均。训练初期样本少，统计量极不准确：

$$\text{更新项} \propto \frac{m_t}{\sqrt{v_t} + \epsilon}$$

若初期 $v_t$ 极小，分母趋近于 $\epsilon$，更新步长会异常巨大。即使有偏差修正（Bias Correction），极早期的不稳定性仍然存在。（①）

**问题三：陷入糟糕的局部最优**

高维损失函数曲面布满浅坑。无 Warmup 时，模型可能因第一步步子太大，直接掉进深而窄的"坏坑"，导致模型只能背诵高频词（如反复输出 "I don't know"），无法继续学习。（①）

### 1.2 学习率调度的核心思想

学习率调度的本质是：**让学习率随训练进程动态变化，在需要探索时放大，在需要精细收敛时缩小**。（③+④）

  ------------------------------------------------------------------------------------
  阶段              学习率特征              目的
  ----------------- ----------------------- --------------------------------------------
  **Warmup 预热**   从极小值线性/余弦增长   保护初期不稳定阶段，让动量统计量"热身"
  
  **Decay 衰减**    从峰值平滑下降         精细收敛，避免在最优解附近震荡
  
  **稳定期**        保持极小值或常数        微调最后几层，确保充分收敛
  ------------------------------------------------------------------------------------

> **经典佐证**：Transformer 原论文《Attention Is All You Need》明确指出："We use the same learning rate schedule as in the original paper: linearly increase for the first $warmup\_steps$, then decrease proportionally." 实验表明，在特定条件下，**无 Warmup 的 Transformer 训练可能极不稳定**。（①）

## 二、Warmup：给模型一个"热身运动"

### 2.1 物理直觉

Warmup 源自体育和机械工程：（①）

- **运动员赛前**：身体冷时肌肉僵硬，直接全力冲刺易拉伤；只有心率提升后，才能进入最佳状态
- **引擎冷启动**：润滑油粘稠，直接高转速会磨损机器；需怠速运转让温度上来

在深度学习中，模型参数从随机初始化开始，对数据一无所知。Warmup 让模型从"冷"平稳过渡到"热"，防止其在未站稳时就"摔跟头"。（①）

### 2.2 线性 Warmup 的数学公式

最常用的线性预热公式为：（①）

$$\eta_t = \eta_{base} \times \min\left(1, \frac{t}{T_{warmup}}\right)$$

其中：
- $t$：当前**步数**（Step，即 batch iteration，不是 epoch）
- $T_{warmup}$：预热总步数
- $\eta_{base}$：基准学习率

**关键理解**：公式中的 $t$ 和 $T_{warmup}$ 单位必须是 **Step（迭代次数）**，不是 Epoch。（①）

> **换算关系**：
> $$\text{总步数} = \text{Epoch 数} \times \left\lceil \frac{\text{数据集大小}}{\text{Batch Size}} \right\rceil$$
>
> 例如：数据集 10,000 条，Batch Size 32，则每 epoch 313 步。若预热 5 个 epoch，则 $T_{warmup} = 5 \times 313 = 1565$ 步。

### 2.3 Transformer 原版公式（Inverse Square Root）

Transformer 论文使用了更精巧的"一条公式覆盖全程"策略：（①）

$$\eta_t = d_{model}^{-0.5} \times \min\left(t^{-0.5},\ t \cdot T_{warmup}^{-1.5}\right)$$

**拆解理解**：

- **$d_{model}^{-0.5}$**：模型维度的缩放系数（如 768 维时约为 $1/\sqrt{768} \approx 0.036$）
- **$t^{-0.5} = 1/\sqrt{t}$**：衰减曲线，随步数增加而下降
- **$t \cdot T_{warmup}^{-1.5}$**：增长直线，随步数线性增加
- **$\min(\cdot, \cdot)$**：取两者较小值，实现自动切换

**两条曲线的交汇**：
- 当 $t < T_{warmup}$：直线值较小，学习率**线性增长**（Warmup）
- 当 $t > T_{warmup}$：曲线值较小，学习率按 $1/\sqrt{t}$ **衰减**

在 $t = T_{warmup}$ 时，两者恰好相等，实现**平滑衔接**。（①）

### 2.4 余弦 Warmup（更平滑的启动）

比线性更温柔的启动方式：（①）

$$\eta_t = \eta_{base} \times \frac{1 - \cos\left(\pi \cdot \frac{t}{T_{warmup}}\right)}{2}$$

- $t=0$ 时导数为 0，启动极其温柔
- 适合对初始化极度敏感的超大模型

## 三、余弦退火：平滑衰减的艺术

### 3.1 为什么选余弦退火？

训练后期，模型已接近最优解，需要**精细调整**。如果学习率下降过于突兀（如 StepLR 直接乘以 0.1），参数会在最优解附近剧烈震荡；如果下降太慢，又会浪费训练时间。（②+④）

余弦退火利用余弦函数在 $[0, \pi]$ 区间从 1 平滑下降到 -1 的特性，让学习率像"波浪"一样优雅地衰减：（②）

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}} \cdot \pi\right)\right)$$

其中：
- $\eta_{max}$：当前周期的最大学习率（即 Warmup 结束时的学习率）
- $\eta_{min}$：最小学习率（通常设为 0 或 $\eta_{base} \times 0.1$）
- $T_{cur}$：当前周期内已进行的步数
- $T_{max}$：半个周期的长度（总衰减步数）

**视觉直觉**：（②）

```
学习率
  ↑    /\
  |   /  \
  |  /    \
  | /      \
  |/        \
  +----------→ 步数
```

- **高学习率阶段**：促进探索，帮助跳出局部最优
- **低学习率阶段**：精细调优，稳定收敛到谷底

### 3.2 与其他衰减策略的对比

PyTorch 提供了多种学习率调度器，各有适用场景：（③+④）

  ---------------------------------------------------------------------------------------
  调度器                    衰减方式                    适用场景
  ------------------------- --------------------------- -----------------------------------
  **StepLR**                每隔固定 epoch 乘以        已知最佳调整时间点的传统 CV 任务
                            gamma                       

  **MultiStepLR**           在指定 milestone 处衰减     需要分阶段精细调整的老式训练流程

  **ExponentialLR**         指数衰减                    需要持续平滑下降的简单任务

  **ReduceLROnPlateau**     基于验证指标自动衰减        不确定何时该降学习率时的保底方案

  **CosineAnnealingLR**     余弦平滑衰减                现代大模型训练的常用选择之一（②）

  **LambdaLR**              自定义函数                  需要特殊曲线（如多项式衰减）时
  ---------------------------------------------------------------------------------------

> **选型建议**：对于 Transformer 和 LLM 训练，**Warmup + CosineAnnealingLR** 是较为常用的组合之一，在特定条件下效果通常较好。StepLR 等离散衰减策略在某些大模型训练场景中可能不如余弦退火平滑。（②）

## 四、PyTorch 工程实践：SequentialLR

PyTorch 1.9+ 提供了 `SequentialLR`，可以像拼积木一样把 Warmup 和衰减串联起来。这是工业界的常见写法。（①）

### 4.1 核心 API 详解

**`LinearLR`**：负责 Warmup 阶段

```python
torch.optim.lr_scheduler.LinearLR(
    optimizer,           # [必填] 优化器实例
    start_factor=1e-8,   # 起始系数，设为极小正数（如 1e-8）模拟从 0 开始
    end_factor=1.0,      # 结束系数，预热结束时达到 1.0 × base_lr
    total_iters=...,     # [关键] 预热总步数（Step，不是 Epoch！）
    last_epoch=-1        # 断点续训时使用
)
```

> **⚠️ 兼容性提示**：在某些 PyTorch 版本中，`start_factor=0.0` 可能引发除零错误或警告。使用 `1e-8` 等极小正数通常更安全，效果等价。如需严格从 0 开始，也可使用 `LambdaLR` 自定义实现。

**`CosineAnnealingLR`**：负责衰减阶段

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=...,       # 衰减阶段的总步数
    eta_min=0.0      # 最小学习率
)
```

**`SequentialLR`**：串联调度器

```python
torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, decay_scheduler],
    milestones=[warmup_steps],  # 在第 warmup_steps 步切换
    last_epoch=-1
)
```

> **⚠️ 关键陷阱**：`milestones` 传入的是**累积步数**。如果 Warmup 占 1000 步，就写 `[1000]`，表示前 1000 步用第一个调度器，之后用第二个。（①）

### 4.2 训练模板

以下代码可直接用于实际项目：（①+②）

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ========== 假设已有 model, train_loader, criterion ==========
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 配置参数
epochs = 100
warmup_epochs = 5
steps_per_epoch = len(train_loader)

# 【关键】所有调度器参数都必须转换为"步数"
warmup_steps = warmup_epochs * steps_per_epoch
total_steps = epochs * steps_per_epoch
decay_steps = total_steps - warmup_steps

# 1. Warmup：从极小值线性增长到 base_lr
scheduler_warmup = LinearLR(
    optimizer,
    start_factor=1e-8,
    end_factor=1.0,
    total_iters=warmup_steps
)

# 2. Decay：余弦退火到 0
scheduler_decay = CosineAnnealingLR(
    optimizer,
    T_max=decay_steps,
    eta_min=0.0
)

# 3. 串联
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_decay],
    milestones=[warmup_steps]
)

# ========== 训练循环 ==========
model.train()
for epoch in range(epochs):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 前向、反向、更新
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 【关键】每个 batch 结束后调用一次！
        scheduler.step()

        # 监控（可选）
        if batch_idx == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1} | LR: {current_lr:.2e}")
```

> **本地运行配置要求**：上述代码在 CPU 或单卡 GPU 上均可运行，内存 ≥ 4GB。

### 4.3 关键陷阱与正确做法

| 陷阱 | 错误做法 | 正确做法 |
|------|----------|----------|
| **单位混淆** | `total_iters=5`（以为是 5 个 epoch） | `total_iters = warmup_epochs * len(train_loader)`，单位永远是 Step（①） |
| **调用时机错误** | `scheduler.step()` 放在 epoch 循环外 | 必须放在 **batch 循环内**，`optimizer.step()` 之后（①） |
| **milestones 理解错误** | 传入相对步数 `[1000, 2000]` | 传入**累积步数**：Warmup 1000 步 + Step 2000 步 = `[1000, 3000]`（①） |
| **重复调用** | epoch 和 batch 里都 `step()` | 只在一个地方调用。基于 Step 的调度器只在 batch 循环内调用（①） |

## 五、与 AMP + 梯度累积的配合

学习率调度与前面学过的进阶技巧完全兼容，但有几个细节需要注意：（①）

### 5.1 与梯度累积的配合

梯度累积改变了"参数更新频率"。基于优化步数设计的调度器（如 `LinearLR`、`CosineAnnealingLR`）应与 `optimizer.step()` 对齐，在真正执行参数更新后调用 `scheduler.step()`。

```python
from torch import amp

scaler = amp.GradScaler()
accumulation_steps = 4

for i, (inputs, labels) in enumerate(train_loader):
    with amp.autocast(device_type='cuda'):
        outputs = model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps
    
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 学习率调度：与参数更新对齐，每 accumulation_steps 步调用一次
        scheduler.step()
```

> **原因**：基于优化步数设计的调度器通常与参数更新次数对齐。梯度累积时，虽然每步都见到了新数据，但参数实际每 `accumulation_steps` 步才更新一次。如果每个 micro-batch 都调用 `scheduler.step()`，当 `accumulation_steps=4` 时，学习率变化速度会变为预期的 4 倍。（①）

### 5.2 与 AMP 的配合

AMP 的 `GradScaler` 和学习率调度器是**相互独立**的两个系统：
- `GradScaler` 管理**损失缩放因子**（解决 FP16 下溢）
- `Scheduler` 管理**学习率**（优化参数更新步长）

两者互不干扰，可以同时使用。

## 六、参数调优指南

### 6.1 Warmup 时长选择

根据底稿①的经验法则：

| 场景 | 推荐 Warmup 步数 | 说明 |
|------|------------------|------|
| 小数据集（< 10万条） | 2~5 个 epoch | 数据有限，过长 warmup 浪费训练步数 |
| 大数据集/大模型预训练 | 总步数的 5%~10% | 通常 2000~10000 步 |
| 微调（Fine-tuning） | 1~3 个 epoch 或 总步数的 3% | 模型已预训练，无需长时间热身 |

### 6.2 基准学习率选择

| 优化器 | 典型 base_lr | 备注 |
|--------|-------------|------|
| AdamW | 1e-4 ~ 5e-4 | Transformer 预训练常用 |
| SGD + Momentum | 1e-2 ~ 1e-1 | 需要更大学习率 |
| 配合 AMP 时 | 与 FP32 相同或略低 | AMP 不影响学习率量级 |

### 6.3 学习率与 batch size 的联动

当通过梯度累积增大等效 batch size 时，学习率通常需要同步调整。经验法则（Linear Scaling Rule）：（①）

$$\eta_{new} = \eta_{base} \times \frac{\text{batch\_size}_{new}}{\text{batch\_size}_{base}}$$

例如：base batch_size=32 时 lr=1e-4，等效 batch_size=128 时，lr 应调整为 $4 \times 10^{-4}$。

> **注意**：此规则在 batch size < 256 时效果通常较好。超大 batch 需配合 Warmup 使用，否则初期仍可能不稳定。

## 七、手动实现（理解原理）

如果你想彻底理解调度器内部逻辑，以下是线性 Warmup + 余弦衰减的手动实现：（①）

```python
import math

def get_lr(step, warmup_steps, total_steps, base_lr):
    """手动计算当前步的学习率"""
    if step < warmup_steps:
        # Warmup：线性增长
        return base_lr * (step / warmup_steps)
    else:
        # 余弦退火
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# 训练循环中使用
current_step = 0
for epoch in range(epochs):
    for batch in train_loader:
        current_step += 1
        lr = get_lr(current_step, warmup_steps, total_steps, base_lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ... 训练代码 ...
```

> **工程建议**：理解手动实现有助于调试，但实际项目中请使用 PyTorch 官方 API（`SequentialLR`），避免边界条件错误和续训状态管理问题。（①）

## 八、常见问题与解决（①+②+③）

  ---------------------------------------------------------------------------------------------------------------------------
  问题                        可能原因                        解决方案
  --------------------------- ------------------------------- -----------------------------------------------------------
  "Loss 在训练初期剧烈震荡或   缺少 Warmup，学习率过大导致     添加 Warmup（至少 3~5 个 epoch），初始学习率从极小值
  变 NaN"                     梯度爆炸                        线性增长（①）

  "Warmup 结束后 Loss 不降     Warmup 时长不够，动量统计量     延长 Warmup 到总步数的 10%；或降低 base_lr
  反而升"                      未稳定                          （①）

  "scheduler.step() 放在       学习率按 epoch 阶梯变化，失去    移到 batch 循环内，每个 batch 调用一次（①）
  epoch 循环里"                Warmup 的平滑保护意义           

  "total_iters 设了 5 个 epoch  混淆了 epoch 和 step            `total_iters = warmup_epochs * len(dataloader)`（①）
  的值但只预热了 5 个 batch"                                  

  "余弦退火后期收敛极慢"        eta_min 设得太高或 T_max 过大   eta_min 设为 0 或 base_lr 的 1%；T_max 等于衰减阶段
                                                              总步数（②）

  "与 DDP 分布式训练配合时      各进程 step 计数不一致           确保 `scheduler.step()` 在每个进程都执行，且
  学习率不同步"                                                dataloader 长度一致（③）

  "断点续训后学习率跳变"        未恢复 scheduler 状态            保存 checkpoint 时同时保存
                                                              `scheduler.state_dict()`，加载时恢复（①）
  ---------------------------------------------------------------------------------------------------------------------------

## 九、总结与后续学习路线（①+②）

恭喜你完成了 PyTorch 进阶训练技巧板块的全部内容！通过本节学习，你应该掌握了：

- ✅ **为什么需要学习率调度**：固定学习率在 Transformer 上会导致梯度爆炸、动量偏差和糟糕局部最优
- ✅ **Warmup 原理**：从"冷启动"到"热热身"，线性/余弦增长保护训练初期稳定性
- ✅ **余弦退火优势**：相比 StepLR 的断崖式下降，余弦曲线提供更平滑的收敛路径
- ✅ **PyTorch 工程实现**：`LinearLR` + `CosineAnnealingLR` + `SequentialLR` 的常见组合
- ✅ **关键陷阱规避**：step 与 epoch 的单位区分、`scheduler.step()` 的正确调用位置
- ✅ **与 AMP/梯度累积的配合**：调度器独立于其他技巧，与 `optimizer.step()` 对齐

> **下一步预告**：在下一板块 **"大语言模型（LLM）核心特性"** 中，我们将跳出技术实现细节，从更高维度理解现代大模型的能力边界——什么是**涌现能力**？为什么大模型能进行**上下文学习（In-Context Learning）**？**指令遵循**和**逐步推理**是如何实现的？以及我们必须正视的**幻觉问题**。现在，试着修改上面的代码，将 Warmup 策略分别改为"线性"和"余弦"，绘制学习率曲线并对比两者的差异！