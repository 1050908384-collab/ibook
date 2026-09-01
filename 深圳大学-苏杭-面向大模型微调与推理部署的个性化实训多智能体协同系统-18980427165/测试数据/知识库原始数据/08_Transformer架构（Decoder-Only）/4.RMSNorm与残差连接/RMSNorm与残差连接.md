# RMSNorm 与残差连接

权威文章：《Attention Is All You Need》  
　　网址：https://arxiv.org/abs/1706.03762

权威文章：《Root Mean Square Layer Normalization》  
　　网址：https://arxiv.org/abs/1910.07467

权威文章：《LLaMA: Open and Efficient Foundation Language Models》  
　　网址：https://arxiv.org/abs/2302.13971

① 底稿：《Transformer 核心必读》规范化层（下）：RMSNorm 详解、数学原理、rsqrt 性能提升、精度转换计算、代码  
　　网址：https://blog.csdn.net/qq_62634342/article/details/159915214

② 参考文章：《Transformer 残差连接原理与优化实践》  
　　网址：https://blog.csdn.net/weixin_29058331/article/details/163146327

> **适合人群**：已完成"重点理解 Decoder-Only 架构"章节，理解 GPT / Llama / Qwen 基本结构的初学者。本节时长约 0.5 小时，目标是理解 RMSNorm 与 LayerNorm 的差异、Pre-Norm / Post-Norm 的区别、残差连接在深层网络中的作用，能够手写 RMSNorm 代码并理解混合精度训练中的类型转换逻辑。（①）
>
> **前置说明**：上节我们梳理了 GPT、Llama、Qwen 三条 Decoder-Only 演进主线，知道了现代 LLM 普遍采用 Pre-RMSNorm、RoPE、SwiGLU 等组件。本节将深入第一个组件——**RMSNorm 与残差连接**，回答：为什么 Llama 要抛弃 LayerNorm？Pre-Norm 和 Post-Norm 到底差在哪？残差连接在 96 层的 GPT-3 中是如何保住梯度不崩的？所有代码请在已安装 `torch` 的 Python 环境中运行。

## 一、深层网络的两大基石：归一化与残差连接

在 Decoder-Only 架构中，GPT-3 有 96 层，Llama、Qwen 等现代模型也有数十层。当网络深度达到这一量级时，两个结构性问题会暴露出来：

1. **梯度消失 / 爆炸**：反向传播时梯度逐层相乘，数值极易失控
2. **训练退化**：深层网络比浅层网络更难训练，甚至表现更差

原始 Transformer 的解决方案是**残差连接（Residual Connection） + 层归一化（Layer Normalization）**。（权威①）

原始 Transformer 中，每个子层的输出为：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

即先通过残差连接将子层输出与输入相加，再做层归一化。这被称为 **Post-LN（Post-LayerNorm）**。（权威①）

而现代 Decoder-Only LLM（如 Llama）普遍采用 **Pre-LN（Pre-LayerNorm）**：

$$x + \text{Sublayer}(\text{RMSNorm}(x))$$

即先对输入做 RMSNorm，再送入子层，最后做残差相加。（权威③ + ②）

本节将拆解：为什么从 LayerNorm 走向 RMSNorm？为什么从 Post-LN 走向 Pre-LN？残差连接究竟在做什么？

## 二、从 LayerNorm 到 RMSNorm

### 2.1 LayerNorm 的原始设计

LayerNorm 对每个样本的所有特征维度计算均值 $\mu$ 和方差 $\sigma^2$，然后标准化：

$$\text{LN}(x_i) = \gamma \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \quad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$$

$\gamma$ 和 $\beta$ 是可学习的缩放和平移参数。（权威①）

### 2.2 RMSNorm 的简化

RMSNorm 论文（权威②）提出一个核心假设：**LayerNorm 的成功主要来自于 re-scaling invariance（重缩放不变性），而非 re-centering invariance（重中心化不变性）。**

因此，RMSNorm 直接去掉了均值计算和偏置 $\beta$，仅保留 RMS（Root Mean Square）统计量：

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$$

$$\text{RMSNorm}(x_i) = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i$$

在数值稳定形式下：

$$\text{RMSNorm}(x_i) = \frac{\gamma_i \cdot x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n} x_j^2 + \epsilon}}$$

（权威② + ①）

### 2.3 为什么可以去掉均值和偏置？

在 **Pre-LN** 架构中，每层结构为 $x + \text{Sublayer}(\text{Norm}(x))$。即使 Norm 层将输入均值变为 0，经过 Sublayer（如 Attention 或 FFN）的线性变换与非线性激活后，输出均值通常不再为 0。当这个输出与原始输入 $x$ 做残差相加时，最终结果的均值主要由 $x$ 决定，**通常不再保持零均值**。（①）

因此：

- **偏置 $\beta$ 是冗余的**：Norm 层的 $\beta$ 与后续线性层的 `bias` 功能重叠，属于参数冗余
- **去均值是冗余的**：残差连接注定会破坏"零均值"特性，在进入 Sublayer 之前费力计算均值并强行减去（LayerNorm 的做法），在下一轮迭代中又会被重置，在特定条件下可能是一种计算上的冗余（①）

RMSNorm 论文的实验表明，在测试条件下，RMSNorm 在 Transformer 等架构上的收敛速度和最终精度与 LayerNorm 通常相近，在某些特定配置的超深网络中可能表现更好，因为它减少了浮点运算带来的累积误差（权威②）。

### 2.4 计算效率对比

假设隐藏维度为 $d$，归一化层作用于每一个 token 的每一个隐藏维度，计算频率极高：

| 步骤 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 求均值 / 平方 | $d$ 次加法算 $\mu$ | $d$ 次乘法算 $x_i^2$ |
| 中心化 / 求和 | $d$ 次减法 $(x_i - \mu)$ | $d$ 次加法算 $\sum x_i^2$ |
| 平方 / 开方 | $d$ 次乘法算方差 | 开方 + 除法 |
| 归一化 | 除法 | 乘法（配合 rsqrt） |
| 仿射变换 | 乘 $\gamma$ 加 $\beta$（$2d$ 次） | 乘 $\gamma$（$d$ 次） |
| **总操作数** | 约 $7d$ 次基础运算 | 约 $4d$ 次基础运算 |

在上述简化分析中，RMSNorm 相比 LayerNorm 减少了部分算术指令（具体比例取决于实现细节和硬件架构），且流程更短，有助于减少 GPU 显存带宽压力。RMSNorm 论文（权威②）报告在特定测试条件下实现了约 **7% ~ 64%** 的速度提升。

## 三、Pre-Norm vs Post-Norm

### 3.1 Post-Norm：原始 Transformer 的选择

原始 Transformer 采用：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

特点：
- 残差分支在归一化之前
- 训练深层网络时稳定性较差，对学习率敏感
- 但模型表达能力理论上更强（②）

### 3.2 Pre-Norm：现代 LLM 的选择

GPT-3、LLaMA、Qwen 等现代模型采用：

$$\text{Output} = x + \text{Sublayer}(\text{RMSNorm}(x))$$

LLaMA 论文明确指出："To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function."（权威③）

特点：
- 归一化在子层输入前，残差连接绕过的是"归一化后的子层"
- 训练通常更稳定，可以使用更大的学习率
- 在现代开源大模型（如 Llama、Qwen 等）中得到了广泛应用（②）

### 3.3 为什么 Pre-LN 与 RMSNorm 较为契合？

Pre-LN 架构中，残差连接直接绕过原始输入 $x$。RMSNorm"只控幅度、不控中心"的策略，与 Pre-LN 的残差逻辑较为契合，避免了 LayerNorm 在残差结构中反复"拉伸 - 还原"均值的冗余计算。（①）

## 四、残差连接的本质

### 4.1 梯度高速公路

没有残差连接时，梯度需要通过链式法则逐层反向传播。在 100 层的网络中，即使每层梯度仅衰减到 0.99，最终梯度也会衰减到 $0.99^{100} \approx 0.366$。残差连接创造了"高速公路"，使梯度可以直接通过短路路径回传，有助于缓解梯度消失。（②）

### 4.2 学习残差而非完整映射

传统网络需要直接学习映射 $F(x)$。残差网络只需学习残差 $H(x) = F(x) - x$。当变换接近恒等映射时，残差趋近于 0，这通常比学习一个任意复杂映射更容易。（②）

### 4.3 残差连接与激活值控制

残差连接的主要作用是改善信息和梯度传播：它为梯度提供了"高速公路"，使深层网络中的梯度可以直接通过短路路径回传，有助于缓解梯度消失问题。同时，残差结构使网络只需学习残差映射 $H(x) = F(x) - x$，当变换接近恒等映射时，这比学习任意复杂映射更容易。

需要注意的是，未经缩放的残差不断相加可能使激活方差随网络深度增长，现代模型通常通过 Pre-Norm/RMSNorm、合理的权重初始化或残差缩放（如 GPT-3 的 $1/\sqrt{N}$ 缩放）来控制这一问题。（②）

### 4.4 标准 Decoder-Only 层的残差结构

现代 Decoder-Only 模型的每一层包含两个残差连接：

1. **Attention 部分**：
   $$x = x + \text{Attention}(\text{RMSNorm}(x))$$

2. **FFN 部分**：
   $$x = x + \text{FFN}(\text{RMSNorm}(x))$$

> **注意**：这里两次 RMSNorm 是独立的，各有一套 $\gamma$ 参数。

## 五、RMSNorm 的代码实现与优化

### 5.1 `torch.rsqrt`：性能引擎

RMSNorm 的核心计算包含除以 $\sqrt{\text{variance} + \epsilon}$。PyTorch 提供 `torch.rsqrt`（Reciprocal Square Root），直接计算 $1/\sqrt{x}$，通常对应单条硬件指令（如 CUDA 的 `rsqrt.approx.f32`）。

**优化对比**：

```python
# ❌ 慢速写法：先开方，再除法（两次高延迟操作）
denominator = torch.sqrt(variance + eps)
output = x / denominator

# ✅ 快速写法：rsqrt + 乘法（一条专用指令 + 一次快速乘法）
inv_rms = torch.rsqrt(variance + eps)
output = x * inv_rms
```

将"开方 + 除法"转化为"rsqrt + 乘法"，有助于提升吞吐量。在大模型训练中，归一化层被调用数十亿次，这点优化累积起来可带来一定的整体训练速度提升（①）。

### 5.2 混合精度训练中的类型转换

大模型训练通常使用 BF16 或 FP16，但**统计量累加必须在 FP32 下进行**：

1. **防止累加溢出**：长序列（如 32K）的平方和累加，极易超出 BF16/FP16 的表示范围变成 `Inf`
2. **防止精度丢失**：低精度下累加时会出现"大数吃小数"，导致方差计算严重失真
3. **保证 rsqrt 稳定性**：FP32 下倒数平方根的精度更有保障，避免梯度异常

因此标准流程是：**输入(BF16) → 转 FP32 计算统计量 → 归一化 → 转回 BF16 → 乘 $\gamma$**（①）

### 5.3 完整代码实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习参数 gamma，初始化为 1，表示初始时不缩放
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 记录原始精度（如 BF16）和设备
        input_dtype = x.dtype
        
        # 【关键】提升到 FP32 计算统计量，防止累加溢出和精度丢失
        x_fp32 = x.float()
        
        # 计算均方值。keepdim=True 保留维度 [B, S, 1]，用于后续广播到 [B, S, D]
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        
        # rsqrt 优化：一步计算 1 / sqrt(variance + eps)
        inv_rms = torch.rsqrt(variance + self.eps)
        
        # 归一化（仍在 FP32 安全区）
        x_normed = x_fp32 * inv_rms
        
        # 【关键】转回原始精度，保证与后续层（BF16 矩阵乘法）的兼容性
        x_normed = x_normed.to(input_dtype)
        
        # 乘以可学习参数 gamma
        return x_normed * self.weight
```

> **本地运行配置要求**：上述代码纯 CPU 即可运行，内存 ≥ 4GB。若需加载预训练模型进行替换实验，建议 GPU 显存 ≥ 8GB。

### 5.4 与 PyTorch 原生 LayerNorm 对比

```python
import torch
import torch.nn as nn

hidden_dim = 512

# 原生 LayerNorm（含均值、方差、gamma、beta）
ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)

# 自定义 RMSNorm（仅 RMS + gamma）
rms_norm = RMSNorm(hidden_dim)

# 模拟输入：[batch=2, seq_len=10, hidden_dim=512]
x = torch.randn(2, 10, hidden_dim)

out_ln = ln(x)
out_rms = rms_norm(x)

print(f"LayerNorm 输出形状: {out_ln.shape}")
print(f"RMSNorm 输出形状:  {out_rms.shape}")

# 观察 gamma 参数量差异
print(f"LayerNorm 参数: {sum(p.numel() for p in ln.parameters())}")   # 1024 (gamma + beta)
print(f"RMSNorm 参数:   {sum(p.numel() for p in rms_norm.parameters())}")  # 512 (仅 gamma)
```

> **观察**：RMSNorm 比 LayerNorm 少了一半的可学习参数（去掉了 $\beta$），且计算路径更短。

## 六、常见问题与解决（① + ②）

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| "去掉均值后效果不会变差吗？" | 忽略了 Pre-LN + 残差的结构特性 | Pre-LN 中残差连接会重置均值，去均值在特定条件下可能是冗余计算；实验表明在测试条件下效果通常相近（①） |
| "Post-LN 和 Pre-LN 怎么选？" | 混淆了架构演进时序 | 现代 LLM 训练通常优先选 Pre-LN，通常更稳定；Post-LN 是原始 Transformer 的选择（②） |
| "代码里为什么要 `.float()`？" | 不理解混合精度训练原理 | 统计量累加在 FP32 下更安全，防止 BF16/FP16 溢出和"大数吃小数"（①） |
| "`rsqrt` 和 `1/sqrt` 有什么区别？" | 不了解硬件指令差异 | `rsqrt` 是单条硬件指令，延迟更低，且将除法转为乘法（①） |
| "训练深层模型 loss 为 NaN" | 梯度爆炸或统计量溢出 | 检查是否使用 Pre-LN + RMSNorm，确认 `eps` 设置合理，添加梯度裁剪（②） |
| "RMSNorm 的 $\gamma$ 初始值为什么是 1？" | 担心初始缩放破坏分布 | 初始值 1 表示"先不缩放"，让网络在训练初期自己学习合适的幅度（①） |

## 七、总结与后续学习路线（权威③ + ①）

恭喜你完成了 RMSNorm 与残差连接的学习！你现在应该掌握了：

- ✅ LayerNorm 与 RMSNorm 的数学差异：RMSNorm 去掉均值 $\mu$ 和偏置 $\beta$，仅保留 RMS 统计量
- ✅ RMSNorm 的理论依据：re-scaling invariance 是 LayerNorm 成功的核心，re-centering 在 Pre-LN 残差结构中可能是冗余的
- ✅ Pre-Norm vs Post-Norm：现代 LLM（Llama、Qwen 等）采用 Pre-RMSNorm 以提升训练稳定性
- ✅ 残差连接的本质：梯度高速公路、学习残差映射，但需注意未经缩放的残差可能导致激活方差增长
- ✅ RMSNorm 的工业级实现：`rsqrt` 硬件级优化、FP32 统计量计算、混合精度类型转换
- ✅ 能够手写 RMSNorm 并理解每一行代码的深意

> **下一步预告**：在下一节 **"前馈网络（FFN）与 SwiGLU 激活函数"** 中，我们将深入讲解 Llama 和 Qwen 为何抛弃 ReLU / GeLU 而选择 SwiGLU，以及 FFN 的维度设计（如 $\frac{2}{3} \cdot 4d$）对模型容量与推理开销的影响。这是理解 Decoder-Only 模型内部计算图的关键。现在，试着修改上面的 RMSNorm 代码，对比在 FP32 和 BF16 下直接计算 `variance` 的差异，观察长序列（如 `seq_len=32768`）时统计量的精度变化！
请审阅，确认无误后请给出下一篇文章。