# 前馈网络（FFN）与 SwiGLU 激活函数

权威文章：《Attention Is All You Need》  
　　网址：https://arxiv.org/abs/1706.03762

权威文章：《GLU Variants Improve Transformer》  
　　网址：https://arxiv.org/abs/2002.05202

权威文章：《LLaMA: Open and Efficient Foundation Language Models》  
　　网址：https://arxiv.org/abs/2302.13971

① 底稿：《深度学习的数学原理（三十一）—— Transformer 前馈网络 FFN》  
　　网址：https://blog.csdn.net/xiaolaji600/article/details/160532839

② 参考文章：《现代 LLM 的核心架构设计其二：SwiGLU》  
　　网址：https://blog.csdn.net/tE6NQPAB2/article/details/163170957

③ 参考文章：《大模型技术解析：从 GLU 到 SwiGLU 的门控机制演进与 Transformer 优化实践》  
　　网址：https://blog.csdn.net/z2a3b4c5d/article/details/155660939

④ 参考文章：《大模型 FFN 中用 SwiGLU 为啥设置 FFN 的映射为 8/3·d 呢？》  
　　网址：https://blog.csdn.net/2501_91504718/article/details/156492861

> **适合人群**：已完成"RMSNorm 与残差连接"章节，理解 Pre-LN、残差连接和 LayerNorm / RMSNorm 差异的初学者。本节时长约 1 小时，目标是理解 FFN 的数学本质、掌握从 ReLU 到 SwiGLU 的演进逻辑、能够推导 SwiGLU 的维度设计原理并手写 SwiGLU FFN 代码。（①）
>
> **前置说明**：上节我们拆解了 RMSNorm 与残差连接，知道了现代 Decoder-Only 模型中每一层的结构是 `x + Sublayer(RMSNorm(x))`。本节将深入 Transformer 层的第二个子层——**前馈网络（FFN）**，回答：为什么 Attention 已经做了全局信息融合，还需要 FFN？为什么 Llama 要抛弃 ReLU 选择 SwiGLU？SwiGLU 的中间维度为什么是 $\frac{8}{3} \cdot d$ 这种奇怪的数字？所有代码请在已安装 `torch` 的 Python 环境中运行。

## 一、FFN 的定位：Attention 的"完美搭档"

在 Decoder-Only 的每一层中，有两个核心子层：

1. **Masked Self-Attention**：让每个 token"看到"序列中其他 token 的信息，解决的是 **"token 之间如何交流"** 的问题。
2. **Feed-Forward Network（FFN）**：对每个 token 独立做非线性特征变换，解决的是 **"每个 token 如何变得更'聪明'"** 的问题。（①）

二者缺一不可：没有 Attention，模型无法捕捉长距离依赖；没有 FFN，模型的特征表达能力会被限制在线性空间，无法拟合复杂的语言模式。（①）

原始 Transformer 中，FFN 的数学定义非常简洁：

$$\text{FFN}(x) = \max\left( 0,xW_{1} + b_{1} \right)W_{2} + b_{2}$$

其中：
- $x \in \mathbb{R}^{L \times d_{\text{model}}}$：输入，$L$ 为序列长度，$d_{\text{model}}$ 为模型维度
- $W_{1} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$：升维矩阵，$d_{\text{ff}} = 4d_{\text{model}}$（如 512 → 2048）
- $W_{2} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$：降维矩阵，保证输出维度与输入一致
- $\max(0, \cdot )$：ReLU 激活函数（权威①）

## 二、原始 FFN 的"升维-激活-降维"三步走

原始 FFN 的前向传播可以清晰拆分为三步，每一步都有明确的数学意义：（①）

### 步骤 1：升维线性变换

$$z_{1} = xW_{1} + b_{1}$$

将每个 token 的 $d_{\text{model}}$ 维特征线性投影到 $d_{\text{ff}}$ 维高维空间。高维空间拥有更多的"特征方向"，为后续非线性变换提供更丰富的特征基础。（①）

### 步骤 2：ReLU 非线性激活

$$z_{2} = \max\left( 0,z_{1} \right)$$

引入非线性，打破"多个线性层叠加仍等价于一个线性层"的局限。ReLU 将负数置 0、保留正数，实现特征的**稀疏激活**。（①）

### 步骤 3：降维线性变换

$$\text{FFN}(x) = z_{2}W_{2} + b_{2}$$

将高维特征投影回 $d_{\text{model}}$ 维，保证 FFN 的输入输出维度一致，使其可以无缝接入残差连接结构。（①）

> **关键理解**：如果没有 ReLU，FFN 的两个线性层可以合并为 $x\left( W_{1}W_{2} \right)$，本质上仍是线性变换，表达能力极其有限。ReLU 的"分段线性"特性让 FFN 可以用多个线性段逼近任意复杂函数。（①）

## 三、从 ReLU 到 SwiGLU：激活函数的演进

### 3.1 ReLU 的局限

ReLU 虽然计算简单，但在深层网络中存在明显缺陷：
- **神经元死亡（Dying ReLU）**：负数区域梯度恒为 0，一旦神经元输出为负，后续训练中将永远不被激活
- **梯度稀疏**：负半轴信息完全丢失，导致梯度流不够平滑（②）

### 3.2 GLU：门控机制的引入

GLU（Gated Linear Unit）最早在 2016 年提出，核心思想是使用**门控机制**动态控制信息流动：

$$\text{GLU}(x) = \sigma\left( xW + b \right) \otimes \left( xV + c \right)$$

其中 $\sigma$ 是 Sigmoid 函数，$\otimes$ 是逐元素乘法。（③）

门控机制就像一个"智能水龙头"：Sigmoid 输出 0~1 之间的门控信号，决定另一条线性路径的信息有多少可以通过。（③）但 Sigmoid 在两端容易饱和（梯度趋近于 0），限制了深层网络的训练效率。（②）

### 3.3 Swish：更平滑的激活

Swish 定义为：

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

当 $\beta = 1$ 时，Swish 在 PyTorch 中对应 `F.silu()`。与 ReLU 相比，Swish 具有以下优势：（②）
- **负半轴非零梯度**：避免了神经元彻底死亡
- **最大梯度约 1.1**：大于 ReLU 的 1.0，反向传播时信号更强
- **平滑非单调**：允许模型对输入进行更灵活的缩放

### 3.4 SwiGLU：门控与 Swish 的结合

SwiGLU 将 GLU 中的 Sigmoid 替换为 Swish，形成 Transformer FFN 的新范式：

$$\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{Swish}(xW) \otimes xV \right)W_{2}$$

其中：
- $xW$ 经过 Swish 激活后作为**门控信号**
- $xV$ 是**值路径**
- 二者逐元素相乘后，再经 $W_{2}$ 投影回原始维度（权威②）

GLU 论文实验表明，SwiGLU 在预训练困惑度和下游任务上均优于 ReLU、GELU 和原始 GLU（权威②）。

## 四、SwiGLU 的维度设计：为什么是 $\frac{8}{3} \cdot d$？

这是面试和工程实现中最常见的问题。核心原因是**参数量守恒**。（④）

### 4.1 原始 FFN 的参数量

原始 FFN 有两个权重矩阵：
- $W_{1} \in \mathbb{R}^{d \times 4d}$
- $W_{2} \in \mathbb{R}^{4d \times d}$

总参数量：
$$4d^{2} + 4d^{2} = 8d^{2}$$

### 4.2 SwiGLU FFN 的参数量

SwiGLU 引入了三组权重矩阵：
- $W \in \mathbb{R}^{d \times d_{\text{ff}}}$（门控路径）
- $V \in \mathbb{R}^{d \times d_{\text{ff}}}$（值路径）
- $W_{2} \in \mathbb{R}^{d_{\text{ff}} \times d}$（输出投影）

总参数量：
$$d \cdot d_{\text{ff}} + d \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d = 3d \cdot d_{\text{ff}}$$

### 4.3 令参数量相等

为了保持与原始 FFN 相近的模型容量和计算量，令：

$$3d \cdot d_{\text{ff}} = 8d^{2}$$

解得：

$$d_{\text{ff}} = \frac{8}{3}d \approx 2.67d$$

这就是 $\frac{8}{3} \cdot d$ 的来源。LLaMA 论文明确表述："We use a dimension of $\frac{2}{3}4d$ instead of $4d$ as in PaLM."（权威③）

### 4.4 工业级对齐细节

实际实现中，$\frac{8}{3}d$ 通常不是整数，需要向上对齐到 **256 的倍数**以优化 GPU 张量核心的计算效率。（④）

以 LLaMA-7B 为例：
- 隐藏维度 $d = 4096$
- 理论中间维度：$4096 \times \frac{8}{3} \approx 10922.67$
- 对齐到 256 的倍数：$\lceil 10922.67 / 256 \rceil \times 256 = 43 \times 256 = 11008$（④）

> **注意**：对齐后的参数量会略微超过原始 FFN，但在工程上这是可接受的 trade-off。

## 五、SwiGLU 的代码实现

### 5.1 标准 SwiGLU FFN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """
    基于 SwiGLU 的 FFN，与 LLaMA 论文对齐
    结构: (SiLU(xW) ⊗ xV) * W2
    """
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()
        # 若未指定 hidden_dim，按 8/3·d 计算并对齐
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x):
        # Swish (SiLU) 门控 + 逐元素乘法 + 降维
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

> **本地运行配置要求**：上述代码纯 CPU 即可运行，内存 ≥ 4GB。若需加载 LLaMA-7B 等预训练模型进行替换实验，建议 GPU 显存 ≥ 16GB。

### 5.2 与原始 ReLU FFN 的参数量对比验证

```python
import torch
import torch.nn as nn

dim = 4096

# 原始 ReLU FFN: d_ff = 4d
relu_ffn = nn.Sequential(
    nn.Linear(dim, 4 * dim),
    nn.ReLU(),
    nn.Linear(4 * dim, dim)
)

# SwiGLU FFN: hidden_dim = 11008 (LLaMA-7B 实际值)
swiglu_ffn = SwiGLUFFN(dim, hidden_dim=11008)

relu_params = sum(p.numel() for p in relu_ffn.parameters())
swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"ReLU FFN 参数量:   {relu_params:,}")      # 约 134.2M
print(f"SwiGLU FFN 参数量: {swiglu_params:,}")    # 约 135.3M
print(f"比值: {swiglu_params / relu_params:.3f}")
```

> **观察**：SwiGLU FFN 的参数量与原始 ReLU FFN 基本持平（比值约 1.008），验证了 $\frac{8}{3}d$ 设计的合理性。

### 5.3 迷你手算实例：验证 SwiGLU 计算流程

为了直观理解 SwiGLU 的计算，我们用极小维度手算一遍：

**设定**：
- $d = 2$，$d_{\text{ff}} = 4$（仅为演示，不按 8/3 比例）
- 输入 $x = [1.0, 2.0]$
- $W = \begin{bmatrix} 0.5 & 1.0 \\ 1.5 & -1.0 \end{bmatrix}$，$V = \begin{bmatrix} -0.5 & 2.0 \\ 0.5 & -2.0 \end{bmatrix}$，$W_{2} = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 1.0 \\ 1.0 & -0.5 \\ -1.0 & 0.5 \end{bmatrix}$

**步骤 1：计算门控和值路径**
$$xW = [1.0, 2.0] \cdot W = [3.5, -1.0]$$
$$xV = [1.0, 2.0] \cdot V = [0.5, -2.0]$$

**步骤 2：Swish 激活门控**
$$\text{Swish}([3.5, -1.0]) = [3.5 \cdot \sigma(3.5),\ -1.0 \cdot \sigma(-1.0)] \approx [3.47, -0.27]$$

**步骤 3：逐元素相乘**
$$[3.47, -0.27] \otimes [0.5, -2.0] = [1.735, 0.54]$$

**步骤 4：降维投影**
$$[1.735, 0.54] \cdot W_{2} = [1.735 \times 0.5 + 0.54 \times (-0.5),\ 1.735 \times 1.0 + 0.54 \times 1.0] = [0.5975, 2.275]$$

可以看到，SwiGLU 通过门控机制对特征进行了动态缩放，负数区域的信息没有被完全丢弃（-0.27 仍保留了部分信号），最终输出与原始输入 $[1.0, 2.0]$ 相比发生了非线性变化。（①+②）

## 六、常见问题与解决（①+②+④）

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| "为什么 Attention 后还要 FFN？" | 混淆了 Attention 与 FFN 的分工 | Attention 负责 token 间信息交互，FFN 负责 per-token 非线性特征变换，二者互补（①） |
| "SwiGLU 为什么要用三个矩阵？" | 不理解门控结构 | 门控路径 + 值路径 + 输出投影，参数量通过 8/3·d 控制（④） |
| "8/3·d 是拍脑袋定的吗？" | 未理解参数量守恒原理 | 令 SwiGLU 三矩阵参数量 3dd_ff 等于原始 FFN 的 8d²，解得 d_ff = 8d/3（④） |
| "Swish 和 ReLU 哪个更快？" | 担心 Swish 引入额外计算 | Swish 仅多一次 sigmoid，在现代 GPU 上开销极小；且 SwiGLU 的 8/3·d 中间维度比 ReLU 的 4d 更小，总计算量基本持平（②） |
| "实际中间维度不是 8/3·d 的精确值？" | 忽略 GPU 对齐要求 | 工程实现中向上对齐到 256 的倍数，如 LLaMA-7B 的 11008（④） |
| "代码里为什么 bias=False？" | 不理解现代 LLM 的简化趋势 | LLaMA、PaLM 等模型在 FFN 中去掉 bias，减少参数量和计算量，不影响性能（权威③） |

## 七、总结与后续学习路线（权威③+①）

恭喜你完成了 FFN 与 SwiGLU 的学习！你现在应该掌握了：

- ✅ FFN 的数学本质：升维 → 非线性激活 → 降维，与 Attention 形成完美互补
- ✅ 从 ReLU 到 SwiGLU 的演进逻辑：ReLU 的神经元死亡问题 → GLU 门控机制 → Swish 平滑激活 → SwiGLU 结合
- ✅ SwiGLU 的维度设计原理：参数量守恒推导，$d_{\text{ff}} = \frac{8}{3}d$
- ✅ 工业级实现细节：256 倍数对齐、bias=False、PyTorch 中 `F.silu()` 即 $\beta=1$ 的 Swish
- ✅ 能够手写 SwiGLU FFN 并理解每一行代码的深意

> **下一步预告**：在下一节 **"旋转位置编码（RoPE）"** 中，我们将深入讲解 Llama 和 Qwen 为何抛弃传统绝对位置编码而选择 RoPE，以及 RoPE 如何通过旋转矩阵将位置信息注入 Attention 的 Q/K 向量中，实现外推性的关键。这是理解 Decoder-Only 模型处理长序列的核心。现在，试着修改上面的 SwiGLU 代码，将 `hidden_dim` 分别设为 $4d$ 和 $\frac{8}{3}d$，观察参数量和前向传播速度的差异！