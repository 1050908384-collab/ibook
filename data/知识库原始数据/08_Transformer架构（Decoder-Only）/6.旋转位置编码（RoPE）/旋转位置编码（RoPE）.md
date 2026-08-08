# 旋转位置编码（RoPE）

权威文章：《RoFormer: Enhanced Transformer with Rotary Position Embedding》  
　　网址：https://arxiv.org/abs/2104.09864

权威文章：《LLaMA: Open and Efficient Foundation Language Models》  
　　网址：https://arxiv.org/abs/2302.13971

① 底稿：《彻底搞懂 RoPE：位置编码的新范式》  
　　网址：https://blog.csdn.net/weixin_53004531/article/details/151040658

② 参考文章：《旋转位置编码（RoPE）详解：从一维序列到三维空间》  
　　网址：https://blog.csdn.net/fydw_715/article/details/147076020

③ 参考文章：《【手撕ROPE】极简方法实现旋转位置编码ROPE，面试遇到手撕ROPE再也不怕了！》  
　　网址：https://blog.csdn.net/qq_30731313/article/details/146071289

④ 参考文章：《RoPE旋转位置编码原理浅析》  
　　网址：https://blog.csdn.net/qq_40672115/article/details/145806475

> **适合人群**：已完成"前馈网络（FFN）与 SwiGLU 激活函数"章节，理解 Decoder-Only 模型中 Attention 和 FFN 分工的初学者。本节时长约 1 小时，目标是理解 RoPE 的数学原理、掌握旋转矩阵的推导逻辑、能够手写 RoPE 代码并理解其相对位置编码的本质。（①）
>
> **前置说明**：上节我们拆解了 FFN 与 SwiGLU，知道了现代 Decoder-Only 模型中每一层的结构是 `x + Attention(RMSNorm(x))` 和 `x + FFN(RMSNorm(x))`。但 Attention 本身是无序的——交换两个 token 的位置，Attention 的输出不变。本节将讲解现代 LLM（Llama、Qwen 等）如何解决这一问题——**旋转位置编码（RoPE）**。回答：为什么不用原始正弦位置编码？RoPE 的"旋转"到底在转什么？为什么说 RoPE 是"用绝对位置编码的方式实现了相对位置编码"？所有代码请在已安装 `torch` 的 Python 环境中运行。

## 一、为什么需要位置编码？

Transformer 的核心是 Self-Attention，而 Attention 的计算本质是**加权平均**：

$$\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

关键问题：**Attention 对输入顺序不敏感**。如果你把序列中第 1 个 token 和第 10 个 token 交换位置，Attention 的输出完全不变。（①）

这意味着，如果没有位置编码，以下两句话对模型来说没有任何区别：

- "从 北京 到 上海 的 火车票"
- "从 上海 到 北京 的 火车票"

因此，必须引入某种机制让模型感知"谁在哪个位置"。（①）

### 1.1 绝对位置编码：原始 Transformer 的选择

原始 Transformer 使用正弦/余弦函数生成位置编码，直接加到词嵌入上：

$$\text{PE}_{(pos,2i)} = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)$$

$$\text{PE}_{(pos,2i + 1)} = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)$$

这种方式简单高效，但存在明显缺陷：（④）
- **外推性差**：训练时最大序列长度 512，推理时 1024 就可能会出问题
- **未显式建模相对位置**：模型需要间接学习"第 5 个词和第 8 个词距离为 3"这一信息

### 1.2 相对位置编码：更自然的思路

语言的本质是**相对位置**而非绝对位置。我们关心的是"这个词距离那个词多远"，而不是"这个词是第 500 个词"。

理想情况下，我们希望 Attention 中的点积满足：

$$\langle f_q(x_m,m), f_k(x_n,n) \rangle = g(x_m, x_n, m - n)$$

即：**Query 和 Key 的点积只依赖于词嵌入和它们的相对距离** $m - n$，与绝对位置 $m, n$ 无关。（权威① + ④）

RoPE 正是实现这一目标的方案。

## 二、RoPE 的核心思想：旋转

### 2.1 从 2D 情况开始理解

假设词嵌入维度 $d = 2$，Query 和 Key 都是二维向量。RoPE 的核心思想是：

> **将词向量按照其位置索引进行旋转，旋转角度与位置成正比。**

具体来说：
- 位置 $m$ 的 Query 向量 $q_m$ 逆时针旋转 $m\theta$ 角度
- 位置 $n$ 的 Key 向量 $k_n$ 逆时针旋转 $n\theta$ 角度

用复数表示（便于理解）：

$$f_q(x_m, m) = (W_q x_m) \cdot e^{im\theta}$$

$$f_k(x_n, n) = (W_k x_n) \cdot e^{in\theta}$$

其中 $e^{i\phi} = \cos\phi + i\sin\phi$ 是单位复数，乘以它等价于旋转 $\phi$ 角度。（权威①）

### 2.2 为什么这样能实现相对位置编码？

我们计算旋转后的 Query 和 Key 的点积（即内积）：

$$\langle f_q(x_m,m), f_k(x_n,n) \rangle = \text{Re}\left[ (W_q x_m)(W_k x_n)^* \cdot e^{i(m - n)\theta} \right]$$

关键观察：**结果中只出现了 $m - n$（相对位置），而 $m$ 和 $n$ 单独消失了！**（权威① + ④）

这意味着：
- 两个 token 相距 5 个位置，无论它们在序列的哪个绝对位置，它们的 Attention 权重模式是相同的
- 这正是"用绝对位置编码的方式实现相对位置编码"的精髓（④）

### 2.3 旋转矩阵的显式形式

将复数表示转换为矩阵形式（与代码一致）。对于二维向量 $(x, y)$，旋转 $\theta$ 角度的变换为：

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

验证：设向量 $X = \rho(\cos\phi, \sin\phi)$，则

$$XR(\theta) = \rho(\cos(\phi + \theta), \sin(\phi + \theta))$$

确实实现了逆时针旋转 $\theta$。（④）

旋转矩阵有两个重要性质，后续推导会用到：
1. **正交性**：$R(\theta)^T = R(-\theta)$（转置等于反向旋转）
2. **可加性**：$R(\theta_1)R(\theta_2) = R(\theta_1 + \theta_2)$（连续旋转等于角度相加）（④）

## 三、从 2D 到高维：分块旋转

实际模型中 $d$ 通常是 4096、5120 等很大的数。RoPE 的处理方式是：

> **将 $d$ 维空间切分为 $d/2$ 个独立的 2D 子空间，每个子空间独立旋转。**

### 3.1 频率参数的定义

对于第 $k$ 个子空间（$k = 0, 1, \ldots, d/2 - 1$），定义频率参数：

$$\theta_k = 10000^{-2k/d}$$

旋转角度为位置索引乘以频率：

$$\text{angle}_{m,k} = m \cdot \theta_k = m \cdot 10000^{-2k/d}$$

这个设置与原始 Transformer 的正弦编码频率一致，确保不同维度的旋转速度呈指数级变化。（权威① + ②）

### 3.2 块对角旋转矩阵

高维旋转矩阵 $R_{\Theta,m}^d$ 是一个块对角矩阵：

$$R_{\Theta,m}^d = \begin{pmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots \\ \sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots \\ 0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots \\ 0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

应用方式：

$$f_q(x_m, m) = R_{\Theta,m}^d \cdot W_q x_m$$

$$f_k(x_n, n) = R_{\Theta,n}^d \cdot W_k x_n$$

由于 $R_{\Theta,m}^d$ 是块对角的，可以高效地逐对计算，无需完整矩阵乘法。（权威①）

### 3.3 高效计算：避免完整矩阵乘法

利用块对角结构的稀疏性，RoPE 可以通过逐元素操作高效实现：

对于向量 $x = (x_1, x_2, x_3, x_4, \ldots, x_{d-1}, x_d)$：

$$\text{RoPE}(x, m) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_2) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_2) \\ \vdots \end{pmatrix}$$

这就是工业实现的核心公式。（权威①）

## 四、RoPE 的关键性质

### 4.1 长距离衰减

RoPE 设置了 $\theta_k = 10000^{-2k/d}$，可以证明：当相对距离 $|m - n|$ 增大时，内积的幅值会衰减。（权威①）

这与我们的直觉一致：**距离越远的 token，关联应该越弱**。RoPE 通过频率设计天然实现了这一特性，无需额外约束。

### 4.2 与线性 Attention 兼容

原始 Attention 的复杂度是 $O(n^2)$。线性 Attention 通过核技巧将其降至 $O(n)$，但大多数相对位置编码无法与之配合。

RoPE 通过旋转注入位置信息，保持了向量范数不变，因此可以自然地与线性 Attention 结合。（权威①）

### 4.3 外推性

由于 RoPE 的旋转角度与位置成连续线性关系，且仅依赖相对距离 $m - n$，模型对未见过的序列长度具有一定的泛化能力。

当然，实际外推效果还受训练数据长度分布影响，这是"长度外推"研究领域的核心话题。（④）

## 五、RoPE 的代码实现

### 5.1 极简版：理解核心逻辑

```python
import torch

def rope(x):
    """
    极简 RoPE 实现
    x: [batch_size, seq_len, head_dim]
    """
    batch_size, seq_len, head_dim = x.shape
    device = x.device
    
    # 计算频率参数 theta_k = 10000^(-2k/head_dim)
    # arange(0, head_dim, 2) 生成 [0, 2, 4, ...]
    theta = 10000 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    
    # 位置索引 [0, 1, 2, ..., seq_len-1]
    pos = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
    
    # 旋转角度: [seq_len, head_dim//2]
    angles = pos * theta
    
    # 计算 cos 和 sin
    cos = torch.cos(angles)  # [seq_len, head_dim//2]
    sin = torch.sin(angles)  # [seq_len, head_dim//2]
    
    # 拆分偶数索引和奇数索引
    x1 = x[..., 0::2]  # [batch, seq_len, head_dim//2]
    x2 = x[..., 1::2]  # [batch, seq_len, head_dim//2]
    
    # 应用 2D 旋转
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # 交错拼接回原始维度
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
```

> **本地运行配置要求**：上述代码纯 CPU 即可运行，内存 ≥ 4GB。

### 5.2 工业级版：支持缓存与广播

实际 LLM 推理中，RoPE 的 cos/sin 值会被反复使用，因此需要缓存。以下是接近 LLaMA 实现的版本：

```python
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    工业级 RoPE 实现，支持缓存和广播
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率参数 theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算并缓存 cos/sin
        self._precompute_cache()
    
    def _precompute_cache(self):
        # 位置索引
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device)
        # 外积: [max_seq_len, dim//2]
        freqs = torch.outer(t, self.inv_freq)
        # 实际存储 cos 和 sin
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))
    
    def forward(self, x, seq_len):
        """
        x: [batch, num_heads, seq_len, head_dim]
        """
        # 取前 seq_len 个缓存值
        cos = self.cos_cached[:seq_len]  # [seq_len, dim//2]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim//2]
        
        # 调整维度以支持广播: [1, 1, seq_len, dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # 拆分
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # 旋转
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
```

### 5.3 在 Attention 中使用 RoPE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=2048)
    
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # 投影到 Q/K/V
        q = self.q_proj(x)  # [batch, seq_len, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 拆分为多头: [batch, num_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE 到 Q 和 K（V 不需要位置编码）
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # 计算 Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]
        
        # Causal Mask（Decoder-Only 必需）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn = attn.masked_fill(mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [batch, heads, seq, head_dim]
        
        # 合并多头并投影输出
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.o_proj(out)
```

> **关键观察**：RoPE 只应用于 **Query 和 Key**，不应用于 Value。原因是 Attention 的权重由 $QK^T$ 决定，位置信息只需要影响"谁关注谁"的权重分配；Value 只负责提供被提取的内容信息，与位置无关。（权威① + ③）

### 5.4 验证：RoPE 的相对位置特性

我们通过一个小实验验证 RoPE 的核心性质：相同相对距离的点积应当相同，与绝对位置无关。

```python
import torch

def test_rope_relative_property():
    """验证 RoPE 的相对位置特性"""
    head_dim = 64
    
    # 构造两个相同的词向量，放在不同绝对位置
    x = torch.randn(1, 1, 1, head_dim)  # [batch=1, heads=1, seq=1, dim]
    
    # 频率参数
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    def rotate_at_pos(vec, pos_idx):
        angles = torch.tensor([[pos_idx]]) * theta  # [1, head_dim//2]
        cos, sin = torch.cos(angles), torch.sin(angles)
        x1, x2 = vec[..., 0::2], vec[..., 1::2]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        return torch.cat([r1, r2], dim=-1)
    
    # 测试相对距离为 5 的两组绝对位置
    r5 = rotate_at_pos(x, 5)
    r10 = rotate_at_pos(x, 10)
    dot_5_10 = (r5 * r10).sum().item()
    
    r20 = rotate_at_pos(x, 20)
    r25 = rotate_at_pos(x, 25)
    dot_20_25 = (r20 * r25).sum().item()
    
    print(f"位置(5,10)的点积: {dot_5_10:.6f}")
    print(f"位置(20,25)的点积: {dot_20_25:.6f}")
    print(f"差值: {abs(dot_5_10 - dot_20_25):.6f}")
    print("差值接近0，验证了相对位置特性！")

test_rope_relative_property()
```

> **本地运行配置要求**：纯 CPU 即可运行，内存 ≥ 4GB。

### 5.5 长距离衰减可视化

RoPE 的一个优雅性质是：随着相对距离增大，内积幅值自然衰减。我们可以通过可视化验证这一点：（权威①）

```python
import torch
import matplotlib.pyplot as plt

def plot_rope_decay(head_dim=64, max_dist=512):
    """可视化 RoPE 的长距离衰减特性"""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # 固定一个单位向量在位置 0
    x = torch.ones(1, head_dim // 2)
    y = torch.ones(1, head_dim // 2)
    
    dots = []
    distances = range(0, max_dist, 8)
    
    for dist in distances:
        angles_0 = torch.zeros(1, head_dim // 2)
        angles_d = torch.tensor([[dist]]) * theta
        
        cos_0, sin_0 = torch.cos(angles_0), torch.sin(angles_0)
        cos_d, sin_d = torch.cos(angles_d), torch.sin(angles_d)
        
        # 旋转后的向量
        r0_1 = x * cos_0 - y * sin_0
        r0_2 = x * sin_0 + y * cos_0
        
        rd_1 = x * cos_d - y * sin_d
        rd_2 = x * sin_d + y * cos_d
        
        # 计算点积（取实部）
        dot = (r0_1 * rd_1 + r0_2 * rd_2).sum().item()
        dots.append(dot)
    
    plt.figure(figsize=(10, 4))
    plt.plot(distances, dots, 'b-', linewidth=2)
    plt.xlabel('Relative Distance |m - n|', fontsize=12)
    plt.ylabel('Inner Product', fontsize=12)
    plt.title('RoPE Long-Term Decay Property', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_rope_decay()
```

> **观察**：随着相对距离增大，内积值逐渐衰减并呈现振荡特性。这与我们的直觉一致——远距离的 token 关联应该更弱，RoPE 通过频率设计天然实现了这一点，无需额外约束。（权威①）

## 六、常见问题与解决（①+②+④）

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| "RoPE 为什么不直接加在 embedding 上？" | 混淆了绝对编码与旋转编码的注入位置 | RoPE 通过旋转 Q/K 实现相对位置建模；若加在 embedding 上则退化为绝对编码，无法保证 $Q_m^T K_n$ 仅依赖 $m-n$（①） |
| "RoPE 只旋转 Q/K，V 为什么不旋转？" | 不理解 Attention 中各角色的分工 | Attention 权重由 $QK^T$ 决定，位置信息只需影响"谁关注谁"；V 提供内容信息，与位置无关（③） |
| "旋转矩阵会不会破坏语义信息？" | 担心旋转改变向量含义 | 旋转是正交变换，保持向量范数和内积结构，仅改变"方向"编码位置，不破坏语义（④） |
| "为什么频率参数是 $10000^{-2i/d}$？" | 不理解频率设计的动机 | 与原始 Transformer 正弦编码一致，确保不同维度旋转速度呈指数级变化，覆盖多尺度位置信息（权威①） |
| "RoPE 和正弦位置编码有什么关系？" | 认为二者完全不同 | RoPE 的频率参数与正弦编码相同，但 RoPE 是"乘性"注入（旋转），正弦编码是"加性"注入；RoPE 可视为正弦编码在 Attention 中的自然推广（④） |
| "代码里为什么用 `x[..., 0::2]` 拆分？" | 不理解高维分块旋转的实现 | 将 $d$ 维拆分为 $d/2$ 个 2D 子空间，偶数索引为 x 坐标，奇数索引为 y 坐标，逐对旋转（③） |
| "RoPE 的外推性到底好不好？" | 混淆理论性质与实际效果 | 理论上 RoPE 支持任意长度（角度连续），但实际外推受训练长度分布限制；超长外推需配合 NTK-aware、YaRN 等改进技术（④） |

## 七、总结与后续学习路线（权威①+③）

恭喜你完成了 RoPE 的学习！你现在应该掌握了：

- ✅ **为什么需要位置编码**：Transformer 的 Attention 本身位置无关，必须显式注入顺序信息
- ✅ **RoPE 的核心思想**：通过旋转矩阵将位置信息编码到 Q/K 向量中，实现"绝对旋转，相对感知"
- ✅ **数学推导**：2D 旋转 → 高维分块旋转 → 频率参数设计 → 块对角矩阵的高效实现
- ✅ **关键性质**：长距离衰减、与线性 Attention 兼容、外推性
- ✅ **工业级实现**：缓存 cos/sin、广播机制、在 Multi-Head Attention 中的正确接入方式
- ✅ **能够手写 RoPE 并理解每一行代码的深意**

> **下一步预告**：在下一节 **"KV Cache 机制"** 中，我们将深入讲解 Decoder-Only 模型在推理时如何复用已计算的 Key 和 Value，避免重复计算，从而将生成速度从 $O(n^3)$ 降至 $O(n^2)$。这是理解大模型推理加速的核心。现在，试着修改上面的 `RoPEAttention` 代码，将 `num_heads` 分别设为 8 和 16，观察 RoPE 的 cos/sin 缓存大小变化，并思考为什么 RoPE 的缓存与 batch size 无关！