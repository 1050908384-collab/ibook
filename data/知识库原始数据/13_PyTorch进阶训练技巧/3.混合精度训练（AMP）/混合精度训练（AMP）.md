# 混合精度训练（AMP）

① 底稿：《Pytorch 混合精度训练最佳实践》  
　　网址：https://blog.csdn.net/sjtu_wyy/article/details/149631663

② 参考文章：《PyTorch自动混合精度（AMP）训练指南：提升训练速度与显存效率》  
　　网址：https://blog.csdn.net/zimskybo/article/details/153269911

> **适合人群**：已完成"梯度累积"章节，理解训练循环、反向传播、优化器更新机制的初学者。本节时长约 1.5 小时，目标是深入理解 AMP 的两个核心组件（autocast 和 GradScaler）的底层工作机制，掌握完整的 AMP 训练代码，并能够系统性排查混合精度训练中的 NaN 问题。（①）
>
> **前置说明**：上节我们学习了梯度累积，知道了如何用时间换空间来模拟大 batch。但即使 batch size 降下来了，如果模型参数量巨大（如 7B），FP32 训练仍然可能因显存不足而失败。**混合精度训练（AMP）** 是解决这个问题的一把利器——它让计算用 FP16（2 字节），参数保存用 FP32（4 字节），在保持数值稳定的前提下将显存占用降低约 30%~50%，并将训练速度提升 2~8 倍。本节将深入讲解：autocast 如何判断哪些操作该转 FP16？GradScaler 的"放大-缩小"算法具体怎么工作？为什么你的 AMP 训练会出 NaN？以及如何与梯度累积无缝配合。

## 一、autocast：自动精度切换的幕后逻辑

### 1.1 为什么需要 autocast？

理想情况下，我们希望所有计算都用 FP16——它快、省显存。但 FP16 的动态范围有限（最小正规格化数约 $6.1 \times 10^{-5}$，最大约 $65504$），某些操作在 FP16 下会**数值爆炸**或**下溢为 0**。（①+②）

`torch.cuda.amp.autocast` 是一个上下文管理器，它内部维护了一张**操作-精度映射表**，自动为每个算子选择最安全、最高效的精度：

```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)   # 在此上下文中，PyTorch 自动选择精度
```

### 1.2 哪些操作自动转 FP16？

PyTorch 官方定义了以下操作在 `autocast` 下自动使用 FP16：（①）

| 操作类型 | 具体算子 | 原因 |
|---------|---------|------|
| 线性变换 | `nn.Linear`, `nn.Conv1d/2d/3d` | 计算密集型，Tensor Core 对 FP16 有专门优化 |
| 矩阵乘法 | `torch.mm`, `torch.matmul`, `torch.bmm` | 同上，FP16 速度可达 FP32 的 8 倍 |
| 卷积 | `nn.ConvTranspose`, `nn.GroupNorm`（部分） | 计算密集型 |
| 激活函数 | `nn.ReLU`, `nn.Tanh`, `nn.Sigmoid` | 逐元素操作，FP16 无精度问题 |
| 逐元素运算 | `torch.add`, `torch.mul` | 开销小，FP16 安全 |

### 1.3 哪些操作强制保留 FP16？

以下操作对数值精度极其敏感，PyTorch 强制它们在 FP32 下执行，即使你在 `autocast` 上下文中：（①+②）

| 操作类型 | 具体算子 | 保留 FP32 的原因 |
|---------|---------|----------------|
| 归一化 | `nn.LayerNorm`, `nn.BatchNorm`, `nn.InstanceNorm` | 统计量（均值、方差）在 FP16 下精度不足 |
| 概率分布 | `nn.Softmax`, `nn.LogSoftmax`, `nn.NLLLoss` | 指数运算容易上溢/下溢 |
| 损失函数 | `nn.CrossEntropyLoss`, `nn.BCELoss`, `nn.MSELoss` | 涉及 log、除法等数值敏感操作 |
| 优化器状态 | Adam 的动量、二阶矩 | 参数更新需要高精度避免漂移 |
| 大规模归约 | `torch.sum`, `torch.mean`（大维度） | 累加误差在 FP16 下会累积 |

> **关键理解**：你不需要记住这张表。`autocast` 已经替你做了最优决策。但在调试 NaN 时，知道"哪些操作被强制保留了 FP32"能帮你快速定位问题。

### 1.4 自定义层的精度控制

如果你写了自定义层，想强制某部分代码不受 `autocast` 影响，有三种方法：（①）

**方法 1：局部禁用 autocast**

```python
with autocast():
    # 这部分自动混合精度
    x = self.conv(x)
    
    with autocast(enabled=False):
        # 这部分强制 FP32
        x = self.custom_norm(x.float())  # 注意：输入必须显式转 FP32
```

**方法 2：模型层强制 FP32**

```python
# 将模型的所有归一化层强制设为 FP32
for m in model.modules():
    if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        m.float()  # 参数和缓存都转为 FP32
```

**方法 3：装饰器（适用于函数级别）**

```python
@torch.cuda.amp.custom_fwd
def my_forward(self, x):
    # 自定义前向，autocast 不会自动转换
    return x
```

## 二、GradScaler：动态损失缩放的算法细节

FP16 的梯度下溢（underflow）是 AMP 训练中最隐蔽的问题。GradScaler 通过**动态损失缩放**解决它。（①+②）

### 2.1 为什么梯度会下溢？

FP16 能表示的最小正规格化正数约为 $6.1 \times 10^{-5}$。在深层网络中，梯度经过链式法则逐层传播，值可能变得非常小（如 $10^{-6}$、$10^{-8}$）。这些梯度在 FP16 下会直接变成 0，导致对应参数永远无法更新。（①）

### 2.2 GradScaler 的"放大-缩小"算法

GradScaler 的核心是一个动态调整的**缩放因子** $\sigma$（初始值通常为 $2^{16} = 65536$）：（①）

**步骤 1：放大损失（Forward）**

```python
scaled_loss = loss * σ
```

损失被放大 65536 倍后，反向传播产生的梯度也会被同步放大，远离下溢区域。

**步骤 2：反向传播（Backward）**

```python
scaled_loss.backward()  # 梯度也被放大了 σ 倍
```

**步骤 3：检查与更新（Step）**

`scaler.step(optimizer)` 内部执行以下逻辑：（①）

1. **梯度合法性检查**：遍历所有参数的 `.grad`，检查是否存在 `inf` 或 `NaN`
2. **如果合法**：
   - 先将梯度除以 $\sigma$ 缩回原尺度
   - 然后执行 `optimizer.step()`
   - 连续 `growth_interval` 次（默认 2000 次）未溢出后，$\sigma \leftarrow \sigma \times \text{growth\_factor}$（默认 1.0001，极缓慢增长）
3. **如果不合法**：
   - 跳过本次 `optimizer.step()`（参数不更新）
   - $\sigma \leftarrow \sigma \times \text{backoff\_factor}$（默认 0.5，直接减半）
   - 清空梯度，等待下一次迭代

**步骤 4：调整缩放因子**

```python
scaler.update()  # 执行上述 σ 的调整逻辑
```

### 2.3 缩放因子的动态变化曲线

想象 $\sigma$ 像是一个"试探水位"的浮标：（①）

- 训练初期，梯度通常较大，$\sigma$ 可能稳定在 $2^{16}$ 附近
- 随着训练深入，如果连续 2000 步都没有溢出，$\sigma$ 会缓慢爬升到 $2^{17}$、$2^{18}$...
- 一旦某一步梯度爆炸（如出现 `inf`），$\sigma$ 立即腰斩，训练"保守"一回合，然后重新试探

这种**自适应机制**让 AMP 训练既不会因缩放过小导致下溢，也不会因缩放过大导致溢出。

### 2.4 查看当前缩放因子

```python
scaler = GradScaler()

# 训练过程中随时查看
print(f"当前缩放因子: {scaler.get_scale()}")
```

> **调试提示**：如果你发现 `scaler.get_scale()` 频繁掉到 $2^{10}$ 甚至更低，说明训练非常不稳定，可能是学习率过大或数据中有异常值。

## 三、完整的 AMP 训练循环

将前面学过的所有知识整合为一个标准模板。这是你在实际项目中可以直接使用的代码框架：（①+②）

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# ========== 假设已有 model, optimizer, criterion, train_loader ==========

scaler = GradScaler()
accumulation_steps = 4  # 与梯度累积配合

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        # 1. 前向传播：在 autocast 上下文中自动混合精度
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # 梯度累积修正
        
        # 2. 反向传播：缩放损失后计算梯度
        scaler.scale(loss).backward()
        
        # 3. 每 accumulation_steps 步更新一次参数
        if (i + 1) % accumulation_steps == 0:
            # scaler.step 内部检查梯度合法性，合法才执行 optimizer.step()
            scaler.step(optimizer)
            scaler.update()      # 调整缩放因子
            optimizer.zero_grad()  # 清空梯度，准备下一周期
        
        # 监控（注意：打印的是缩放前的 loss）
        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item() * accumulation_steps:.4f}, "
                  f"Scale: {scaler.get_scale()}")
```

> **本地运行配置要求**：需要 NVIDIA GPU（Volta 架构及以上），显存 ≥ 8GB。CPU 可运行但无加速效果。

### 3.1 代码要点解读

1. **`with autocast()`**：包裹前向传播和损失计算，自动处理精度切换
2. **`loss / accumulation_steps`**：梯度累积的数学修正（见上节）
3. **`scaler.scale(loss).backward()`**：先放大损失，再反向传播
4. **`scaler.step(optimizer)`**：不是简单的 `optimizer.step()`，而是"检查 → 缩放回 → 更新"的原子操作
5. **`scaler.update()`**：调整缩放因子，必须在 `scaler.step()` 之后调用

## 四、AMP + 梯度累积：双剑合璧

在大模型训练中，AMP 和梯度累积几乎总是**同时使用**的。两者的配合逻辑如下：（①）

| 技术 | 解决的问题 | 配合方式 |
|------|-----------|---------|
| AMP | 显存占用高、计算速度慢 | FP16 计算 + FP32 保存 |
| 梯度累积 | batch size 受限 | 多步累加梯度，等效大 batch |

**配合时的注意事项**：

1. **缩放顺序**：先 `loss / accumulation_steps`，再 `scaler.scale(loss)`
   ```python
   loss = criterion(outputs, labels) / accumulation_steps
   scaler.scale(loss).backward()
   ```
   顺序不能反！如果先 `scaler.scale()` 再除以 `accumulation_steps`，GradScaler 的溢出检测会基于错误的梯度尺度。

2. **`scaler.step()` 只在累积满后调用**：不要在每个小 batch 都调用，否则梯度还没累积完就被清空了。

3. **缩放因子的变化频率**：由于 `scaler.update()` 只在累积满后调用，等效于每 $N$ 个 batch 才调整一次 $\sigma$。这是正常的，不影响训练稳定性。

## 五、NaN 问题的系统性排查

AMP 训练中最让人头疼的问题就是 **Loss 突然变成 NaN**。底稿①提供了一套非常实用的四层排查法：（①）

### 5.1 第一层：数据排查

```python
# 检查输入数据是否有异常值
print(f"输入最大值: {inputs.max()}, 最小值: {inputs.min()}")
print(f"输入是否有 NaN: {torch.isnan(inputs).any()}")
print(f"输入是否有 Inf: {torch.isinf(inputs).any()}")
```

**常见问题**：
- 数据预处理时除以了 0
- 归一化参数（mean/std）与数据集不匹配
- 数据加载器返回了损坏的样本

### 5.2 第二层：模型排查

```python
# 检查模型参数初始化是否合理
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"参数 {name} 包含 NaN/Inf")
```

**常见问题**：
- 自定义层的初始化不当（如权重全 0 或极大）
- 预训练模型权重损坏
- 某些层（如 LayerNorm）的 epsilon 设置过小

### 5.3 第三层：训练机制排查

```python
# 临时关闭 AMP，用 FP32 训练一个 batch 看是否正常
with autocast(enabled=False):
    outputs = model(inputs.float())
    loss = criterion(outputs, labels)
```

**判断逻辑**：（①）
- 如果 FP32 正常、FP16 出 NaN → 典型的混合精度问题，继续第四层排查
- 如果 FP32 也出 NaN → 问题与 AMP 无关，检查数据、模型、学习率

### 5.4 第四层：混合精度配置排查

如果确认是 AMP 导致的问题，按以下优先级修复：（①）

**修复 1：降低初始缩放因子**

```python
scaler = GradScaler(init_scale=2.0**10)  # 默认 2^16，降低到 2^10
```

**修复 2：添加梯度裁剪**

```python
scaler.unscale_(optimizer)  # 先取消缩放，让梯度回到原始尺度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

> **注意**：`clip_grad_norm_` 必须在 `scaler.step()` 之前调用，且需要先 `scaler.unscale_(optimizer)` 将梯度缩回真实尺度，否则裁剪阈值对放大后的梯度无效。

**修复 3：强制数值敏感层使用 FP32**

```python
for m in model.modules():
    if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        m.float()
```

**修复 4：降低学习率**

FP16 的梯度对步长更敏感。尝试将学习率降至 FP32 训练时的 1/2 ~ 1/10。

**修复 5：检查损失函数**

某些损失函数（如自定义的带 log 的损失）在 FP16 下容易下溢。确保使用 PyTorch 官方实现，或手动在 `autocast(enabled=False)` 中计算。

## 六、最佳实践总结（①+②）

| 实践 | 说明 |
|------|------|
| **始终使用官方 API** | `torch.cuda.amp.autocast` 和 `GradScaler` 是 PyTorch 官方维护的，不要自行实现 FP16 转换 |
| **不要手动 `.half()` 模型** | `model.half()` 会将所有层强制转 FP16，破坏数值稳定性。让 `autocast` 自动决策 |
| **监控缩放因子** | 定期打印 `scaler.get_scale()`，如果持续下降说明训练不稳定 |
| **配合梯度累积使用** | 大模型训练的标配组合，注意 `loss / accumulation_steps` 在 `scaler.scale()` 之前 |
| **保存 checkpoint 时存 FP32 权重** | `scaler.step()` 更新的是 FP32 参数，直接保存即可 |
| **分布式训练时每个进程独立 scaler** | `GradScaler` 不是线程安全的，每个 GPU 进程应有自己的实例 |

## 七、常见问题与解决（①+②）

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| "Loss 突然变成 NaN" | 缩放因子过大、学习率过高、数据异常 | 按 5.1~5.4 四层排查法系统排查（①） |
| "显存没有明显减少" | 模型过小或激活值不是瓶颈 | AMP 对大型 Transformer/CNN 效果最明显；尝试增大 batch size（②） |
| "速度提升不明显" | GPU 无 Tensor Cores 或模型受 CPU 预处理瓶颈 | 确认 GPU 计算能力 ≥ 7.0；使用 `pin_memory=True` 和 `non_blocking=True`（①） |
| "梯度裁剪后 scaler 报错" | `clip_grad_norm_` 在 `scaler.step()` 之后调用 | 正确顺序：`unscale_` → `clip_grad_norm_` → `step` → `update`（①） |
| "验证/推理时是否需要 autocast？" | 不确定推理阶段是否用 AMP | 验证/推理可以用 `with autocast():`，能加速且不影响精度；但保存结果前确保转回 FP32（②） |
| "分布式训练中 scaler 如何同步？" | 多卡间缩放因子不一致 | 每个进程独立维护 scaler，PyTorch DDP 的 `all_reduce` 只同步梯度，不同步 scaler 状态（①） |

## 八、总结与后续学习路线（①+②）

完成本节学习后，你应该掌握了：

- ✅ **autocast 的自动精度切换策略**：知道哪些操作自动转 FP16，哪些强制保留 FP32
- ✅ **GradScaler 的动态缩放算法**：理解"放大损失 → 反向传播 → 检查合法性 → 缩回更新"的完整流程
- ✅ **完整的 AMP 训练代码**：能够独立编写与梯度累积配合的标准训练循环
- ✅ **NaN 的四层排查法**：从数据 → 模型 → 训练机制 → 混合精度配置，系统性定位问题
- ✅ **最佳实践**：不手动 `.half()`、监控缩放因子、正确顺序使用梯度裁剪

> **下一步预告**：在下一节 **"梯度检查点（Gradient Checkpointing）"** 中，我们将学习另一种"用时间换空间"的技术——通过丢弃中间激活值、反向传播时重算，来进一步降低显存占用。这是与 AMP、梯度累积并列的三大显存优化手段。现在，试着修改上面的代码，在 `autocast` 中故意构造一个数值溢出场景（如输入极大值），观察 `scaler.get_scale()` 如何变化！