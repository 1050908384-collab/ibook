# 掌握梯度累积（Gradient Accumulation）

① 底稿：《显存爆炸解决方法之梯度累积：是什么、为什么、怎么做？从数学原理到代码落地的全流程指南》  
　　网址：https://blog.csdn.net/qq_62634342/article/details/162246553

② 参考文章：《显存优化：梯度检查点 Gradient Checkpoint 和梯度累积》  
　　网址：https://blog.csdn.net/vivi_cin/article/details/147870674

> **适合人群**：已完成"PyTorch 基础"板块和 AMP 环境准备，理解训练循环、优化器、损失函数的初学者。本节时长约 1 小时，目标是理解梯度累积的数学原理、掌握正确的代码写法（尤其是 loss 为什么要除以 accumulation_steps），并能够与 AMP 配合使用。（①）
>
> **前置说明**：上节我们确认了环境支持 AMP。但在实际大模型训练中，还有一个更常见的瓶颈——**显存不足**。假设你的实验表明 batch_size=32 时模型刚好不爆显存，但论文推荐的 batch_size=128 才能收敛到最优效果，怎么办？买更贵的显卡？不，**梯度累积** 可以在不增加显存占用的前提下，用多个小 batch 模拟大 batch 的训练效果。本节将回答：梯度累积的数学原理是什么？为什么代码里必须把 loss 除以 accumulation_steps？它与梯度检查点有什么区别？

## 一、为什么需要梯度累积？

### 1.1 显存瓶颈：batch size 的天花板

在 PyTorch 训练中，显存主要被三部分占用：

1. **模型参数**：与参数量成正比，固定不变
2. **优化器状态**：如 Adam 的动量、二阶矩，与参数量成正比
3. **激活值（Activations）**：前向传播产生的中间结果，**与 batch size 成正比**

其中，激活值是 batch size 的"杀手"。当你把 batch size 从 32 提升到 128 时，激活值显存占用会同步增长 4 倍，很容易触发 OOM（Out Of Memory）。（①）

**梯度累积的核心思想**：将一个大 batch 拆成 $N$ 个小 batch，每个小 batch 单独计算梯度并累加，累积满 $N$ 步后再执行一次参数更新。这样，显存占用始终等于小 batch 的水平，但参数更新的频率和方向与大 batch 一致。（①+②）

### 1.2 梯度累积 vs 梯度检查点

在进阶训练技巧中，有两个名字很像但原理完全不同的技术：（②）

| 特性 | 梯度累积（Gradient Accumulation） | 梯度检查点（Gradient Checkpointing） |
|------|--------------------------------|-----------------------------------|
| **核心思想** | 用时间换 batch size | 用时间换空间 |
| **显存节省来源** | 始终使用小 batch，激活值少 | 丢弃中间激活值，反向传播时重算 |
| **计算开销** | 无额外计算，但更新次数减少 | 在特定条件下可能增加约 30% 计算时间（②） |
| **代码实现** | `accumulation_steps` 参数控制 | `torch.utils.checkpoint` |
| **适用场景** | 需要大 batch 稳定性但显存不足 | 模型极深，激活值是主要瓶颈 |

> **本节聚焦梯度累积**。梯度检查点将在下一节详细讲解。

## 二、数学原理：为什么 loss 必须除以 accumulation_steps？

这是梯度累积中最容易出错、也最关键的问题。让我们从 PyTorch 的默认行为出发，一步步推导。（①）

### 2.1 PyTorch 默认的 reduction='mean'

PyTorch 的损失函数（如 `nn.CrossEntropyLoss`）默认使用 `reduction='mean'`：

$$L_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \ell_i$$

反向传播时，梯度正比于这个平均损失：

$$g_{\text{batch}} \propto \nabla L_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell_i$$

**关键结论**：当使用 `mean` 时，**梯度大小与 batch size 成反比**。（①）

- batch_size = 32 → 梯度 $\propto 1/32$
- batch_size = 128 → 梯度 $\propto 1/128$

在相同数据量下，小 batch 的梯度是大 batch 的 4 倍。

### 2.2 场景分析：模拟 batch_size=128

假设显存只能支持 batch_size=32，但我们需要 batch_size=128 的效果。设 `accumulation_steps = 4`。

**场景 A：直接大 batch（目标）**

一次性输入 128 个样本：

$$L_{\text{target}} = \frac{1}{128} \sum_{i=1}^{128} \ell_i$$

$$G_{\text{target}} \propto \frac{1}{128} \sum_{i=1}^{128} \nabla \ell_i$$

**场景 B：梯度累积，但不缩放 loss（错误做法）**

分 4 次输入，每次 32 个样本。每次的 loss 都是 `mean`，分母为 32：

$$L_{\text{small}}^{(i)} = \frac{1}{32} \sum_{j \in \text{batch}_i} \ell_j$$

每次反向传播得到的梯度：

$$g_i \propto \frac{1}{32} \sum_{j \in \text{batch}_i} \nabla \ell_j$$

PyTorch 的 `backward()` 会**累加**梯度到 `.grad`：

$$G_{\text{accum}} = g_1 + g_2 + g_3 + g_4 \propto \frac{\sum_{i=1}^{128} \nabla \ell_i}{32}$$

对比目标梯度：

$$G_{\text{accum}} = \frac{128}{32} \times G_{\text{target}} = 4 \times G_{\text{target}}$$

**结论**：如果不缩放 loss，累积梯度是目标梯度的 4 倍（即 $N$ 倍）！这会导致参数更新步长过大，训练震荡甚至发散。（①）

**场景 C：正确的做法——每个小 batch 的 loss 除以 $N$**

$$L_{\text{correct}}^{(i)} = \frac{L_{\text{small}}^{(i)}}{N} = \frac{1}{4} \times \frac{1}{32} \sum_{j \in \text{batch}_i} \ell_j = \frac{1}{128} \sum_{j \in \text{batch}_i} \ell_j$$

对应的梯度：

$$g_i' \propto \frac{1}{128} \sum_{j \in \text{batch}_i} \nabla \ell_j$$

累加后：

$$G_{\text{final}} = \sum_{i=1}^{4} g_i' \propto \frac{\sum_{i=1}^{128} \nabla \ell_i}{128} = G_{\text{target}}$$

**数学上等价**！`loss = loss / accumulation_steps` 是梯度累积中通常必不可少的修正。（①）

### 2.3 如果 loss 用 reduction='sum' 呢？

如果你的损失函数设置为 `reduction='sum'`：

$$L_{\text{batch}} = \sum_{i=1}^{B} \ell_i$$

此时梯度：

$$g_{\text{batch}} \propto \sum_{i=1}^{B} \nabla \ell_i$$

分母中没有 $B$，梯度大小与 batch size 无关。因此：

- 小 batch 跑 4 次，每次梯度 $g_i \propto \sum_{32} \nabla \ell_j$
- 累加后 $G_{\text{accum}} = \sum g_i \propto \sum_{128} \nabla \ell_j = G_{\text{target}}$

**结论**：如果使用 `reduction='sum'`，不需要除以 `accumulation_steps`！（①）

> **但注意**：通常情况下我们都用默认的 `reduction='mean'`，因为 mean 让 loss 数值与 batch size 无关，方便监控。所以**请务必记得除以 accumulation_steps**。

### 2.4 物理直觉：batch size 与训练稳定性

除了数学等价性，batch size 还影响训练动态：（①）

| 特性 | 小 batch size | 大 batch size |
|------|--------------|--------------|
| 梯度幅度（未缩放时） | 大 | 小 |
| 梯度方向噪声 | 高（随机震荡） | 低（稳定精准） |
| 训练稳定性 | 低，loss 曲线振荡 | 高，loss 平滑下降 |
| 泛化能力 | 往往更好，能跳出尖锐局部极小 | 可能收敛到尖锐极小，泛化稍差 |
| 收敛速度（每个 epoch） | 更新次数多，但方向不准 | 更新次数少，但方向准 |

**形象比喻**：
- 小 batch 像一群盲人摸象，每个人只摸到一小块，然后各自朝自己以为正确的方向冲，乱哄哄但偶尔能发现新路径。
- 大 batch 像用高清雷达扫描整片区域，然后统一朝着最可靠的方向稳步前进。

梯度累积让我们既能享受"高清雷达"的稳定性，又不需要一次性加载整片区域的数据。（①）

## 三、代码实现

### 3.1 基础版：标准训练循环 vs 梯度累积

**标准训练循环（batch_size=128）**：

```python
for epoch in range(num_epochs):
    for inputs, labels in dataloader:  # 每次 128 个样本
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**梯度累积版（batch_size=32，accumulation_steps=4）**：

```python
accumulation_steps = 4

for epoch in range(num_epochs):
    optimizer.zero_grad()  # 每个 epoch 开始时清零
    accumulated_steps = 0  # 记录当前周期已累积的步数
    
    for i, (inputs, labels) in enumerate(dataloader):  # 每次 32 个样本
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 关键：除以 accumulation_steps
        loss = loss / accumulation_steps
        loss.backward()  # 梯度自动累加到 .grad
        accumulated_steps += 1
        
        # 判断是否为最后一个 batch
        is_last_batch = (i == len(dataloader) - 1)
        
        # 每 accumulation_steps 步更新一次参数；
        # 如果是最后一个 batch 且累积未满，也执行更新，避免丢失尾部梯度
        if accumulated_steps == accumulation_steps or (is_last_batch and accumulated_steps > 0):
            # 如果是不完整周期，补偿梯度以正确归一化
            if is_last_batch and accumulated_steps < accumulation_steps:
                scale_factor = accumulation_steps / accumulated_steps
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)
            
            optimizer.step()
            optimizer.zero_grad()
            accumulated_steps = 0
```

> **本地运行配置要求**：上述代码在 CPU 或单卡 GPU 上均可运行，内存 ≥ 4GB。

### 3.2 关键代码解读

1. **`loss = loss / accumulation_steps`**：这是数学等价性的核心。如果不做这步，累积梯度会过大。
2. **`optimizer.zero_grad()` 的位置**：在每个累积周期开始时清零，而不是每个 batch 都清零。如果每个 batch 都清零，梯度就无法累加。
3. **`is_last_batch` 判断**：通过 `i == len(dataloader) - 1` 检测最后一个 batch，确保即使累积未满也执行参数更新，避免尾部梯度丢失。
4. **不完整周期补偿**：如果最后一个周期只累积了 `accumulated_steps` 步（小于 `accumulation_steps`），由于之前每步都按 `accumulation_steps` 缩放，当前梯度是目标梯度的 `accumulated_steps / accumulation_steps` 倍。通过 `grad.mul_(accumulation_steps / accumulated_steps)` 补偿，使最终梯度等效于按实际步数归一化。
5. **`accumulated_steps` 计数器**：显式追踪当前周期已累积的步数，比 `(i + 1) % accumulation_steps == 0` 更灵活，能优雅处理尾部不完整周期。

### 3.3 与 AMP 混合精度配合

上节我们学习了 AMP，现在将两者结合——这是大模型训练中常见的配置：

```python
from torch import amp

scaler = amp.GradScaler()
accumulation_steps = 4

for epoch in range(num_epochs):
    optimizer.zero_grad()
    accumulated_steps = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        with amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # 缩放损失
        
        # scaler 自动处理 FP16 梯度的缩放和累加
        scaler.scale(loss).backward()
        accumulated_steps += 1
        
        is_last_batch = (i == len(dataloader) - 1)
        
        # 满步更新，或最后一个 batch 时强制更新
        if accumulated_steps == accumulation_steps or (is_last_batch and accumulated_steps > 0):
            # 不完整周期补偿
            if is_last_batch and accumulated_steps < accumulation_steps:
                scale_factor = accumulation_steps / accumulated_steps
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)
            
            scaler.step(optimizer)  # 内部检查梯度合法性后更新
            scaler.update()
            optimizer.zero_grad()
            accumulated_steps = 0
```

> **注意**：`scaler.scale(loss)` 会在 `loss / accumulation_steps` 的基础上再做一次损失缩放（loss scaling），两者互不冲突。`accumulation_steps` 修正的是 batch size 带来的梯度尺度问题，`GradScaler` 修正的是 FP16 的动态范围问题。（①）

## 四、进阶话题

### 4.1 学习率与 batch size 的联动

当 batch size 变化时，学习率通常也需要调整。经验法则（Linear Scaling Rule）：（①）

$$\eta_{\text{new}} = \eta_{\text{base}} \times \frac{\text{batch\_size}_{\text{new}}}{\text{batch\_size}_{\text{base}}}$$

例如：
- 基线：batch_size=32，lr=1e-4
- 目标：等效 batch_size=128（通过梯度累积）
- 新学习率：$1\text{e-4} \times \frac{128}{32} = 4\text{e-4}$

> **注意**：这个规则在 batch size 较小时（<256）效果通常较好。当 batch size 极大时，需要配合 Warmup 使用，否则初期训练不稳定。

### 4.2 临界点理论：batch size 越大越好吗？

部分研究表明（如 ResNet 训练 ImageNet），存在一个**临界 batch size**，在特定条件下可能在 256 或 512 附近：（①）

- **< 临界点**：增大 batch size 通常显著提升速度，对精度影响较小
- **> 临界点**：速度提升变缓，**测试精度可能开始下降**

原因：超大 batch 的梯度方向过于"精准"，模型容易一头扎进**尖锐的局部最优解**（Sharp Minima）。这种解在训练数据上表现较好，但对参数扰动极其敏感，泛化性通常较差。而小 batch 的梯度噪声反而帮助模型落入**平坦的全局最优解**（Flat Minima），泛化性通常更好。（①）

因此，梯度累积虽然能模拟任意大的 batch size，但**不建议将等效 batch size 设置得过大**。通常 64~256 是性价比较高的区间。

## 五、常见问题与解决（①+②）

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| "忘了除以 accumulation_steps，loss 不降反升" | 累积梯度过大，更新步长失控 | 务必添加 `loss = loss / accumulation_steps`（①） |
| "梯度累积后训练速度变慢" | 正常现象，用时间换空间 | 确认 GPU 利用率是否饱和；如果已饱和，这是预期代价（②） |
| "最后一个 epoch 的梯度不完整" | 数据集大小不是 accumulation_steps 的整数倍 | 使用本文 3.1 节的代码：通过 `is_last_batch` 判断强制更新，并对不完整周期进行梯度补偿（①） |
| "与 AMP 配合时出现 NaN" | 双重缩放冲突或学习率过大 | 检查 `loss / accumulation_steps` 是否在 `autocast` 内；降低学习率（①） |
| "不确定等效 batch size 该设多大" | 不了解临界点理论 | 从 64~128 开始实验，观察验证集精度，逐步调整（①） |
| "梯度累积和梯度检查点能一起用吗？" | 混淆两者 | 可以！梯度检查点减少激活值显存，梯度累积模拟大 batch，两者互补（②） |

## 六、总结与后续学习路线（①+②）

完成本节学习后，你应该掌握了：

- ✅ **为什么需要梯度累积**：显存不足时，用小 batch 模拟大 batch 的训练效果
- ✅ **数学原理**：PyTorch 默认 `reduction='mean'` 导致梯度与 batch size 成反比，必须通过 `loss / accumulation_steps` 修正
- ✅ **代码实现**：正确的 `zero_grad()` 位置、`backward()` 累加、`step()` 触发时机，以及**尾部不完整周期的检测与梯度补偿**
- ✅ **与 AMP 配合**：`scaler.scale(loss / accumulation_steps).backward()` 的标准写法，包含尾部补偿
- ✅ **工程权衡**：梯度累积是"用时间换空间"，与梯度检查点互补而非替代

> **下一步预告**：在下一节 **"混合精度训练（AMP）"** 中，我们将深入讲解 `autocast` 和 `GradScaler` 的底层机制，理解 FP16 的动态范围限制、损失缩放的数学原理，以及如何在训练出现 NaN 时进行调试。现在，试着修改上面的代码，将 `accumulation_steps` 分别设为 2、4、8，观察在相同 epoch 数下训练 loss 的收敛曲线差异！