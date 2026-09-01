# Self-Attention 与 Multi-Head Attention：计算原理与公式推导

> **参考文章链接**
> 
> ① 底稿：《Self-Attention机制的计算详解》
> 　　网址：https://blog.csdn.net/qq_41915623/article/details/125161008
> 
> ② 参考文章1：《手写Multi-Head Attention多头注意力机制，Pytorch实现与原理详解》
> 　　网址：https://blog.csdn.net/weixin_62761878/article/details/157441565
> 
> ③ 参考文章2：《〖手撕代码〗多头注意力机制的实现代码是怎么写的？？Multi-Head Attention MHA》
> 　　网址：https://blog.csdn.net/weixin_52336773/article/details/153317897

---

**适合人群**：已掌握 QKV 核心概念，希望深入理解 Self-Attention 与 Multi-Head Attention 的数学公式、维度变化和代码实现的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够手写 Multi-Head Attention 模块，理解每一步的维度变化，为后续"Transformer 架构""大语言模型"打下坚实基础。

**预计学习时长**：约 1.5 小时  
**环境准备**：Python 3.10+，安装 `torch`、`einops`

```bash
pip install torch einops
```

---

## 第一章：Self-Attention 的完整计算流程

### 1.1 核心公式回顾

Self-Attention（自注意力）的数学表达：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q = XW^Q$（Query，查询）
- $K = XW^K$（Key，键）
- $V = XW^V$（Value，值）
- $X$：输入序列的嵌入矩阵
- $W^Q, W^K, W^V$：可学习的线性投影矩阵
- $d_k$：$Q$/$K$ 向量的维度（单头情况下等于 $d_{model}$）

**为什么这样写**：在 Self-Attention 中，$Q$、$K$、$V$ 都由**同一个输入 $X$** 生成，所以叫"自"注意力——自己查自己。

### 1.2 维度变化全解析

假设输入一个句子："我想吃酸菜鱼"，共 6 个字，每个字用 768 维向量表示：

| 张量 | 形状 | 含义 |
|------|------|------|
| $X$（输入） | $[6, 768]$ | 6 个词，每个词 768 维 |
| $W^Q, W^K, W^V$ | $[768, 768]$ | 投影矩阵 |
| $Q, K, V$ | $[6, 768]$ | 投影后的查询/键/值 |
| $QK^T$ | $[6, 6]$ | 注意力分数矩阵 |
| $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ | $[6, 6]$ | 注意力权重矩阵 |
| $\text{Attention}(Q,K,V)$ | $[6, 768]$ | 输出（与输入同维度） |

**为什么这样写**：$QK^T$ 的结果是 $[6, 6]$，表示**每个词对其他每个词的关注程度**。第 $i$ 行第 $j$ 列的值，表示"第 $i$ 个词对第 $j$ 个词的注意力权重"。

### 1.3 四步拆解：从输入到输出

**第一步：线性投影生成 Q、K、V**

```python
import torch
import torch.nn as nn

d_model = 768   # 模型维度
seq_len = 6     # 序列长度（6个字）

# 输入 X: [seq_len, d_model]
X = torch.randn(seq_len, d_model)

# 三个独立的线性投影层
W_Q = nn.Linear(d_model, d_model, bias=False)
W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)

Q = W_Q(X)   # [6, 768]
K = W_K(X)   # [6, 768]
V = W_V(X)   # [6, 768]
```

**为什么这样写**：
- 三个独立的 $W^Q, W^K, W^V$ 让模型从同一个输入中提取三种不同"视角"的信息。
- 关于 `bias=False`：很多**现代 LLM 实现**（如 Llama 系列的注意力投影层）会关闭偏置，以减少参数量和计算开销；但**并非所有 Transformer 都这样**，原始 Transformer 论文的注意力投影是可以带偏置的。实际是否带偏置，取决于具体的模型架构设计，不属于必须遵守的规则。

**第二步：计算注意力分数**

```python
import math

# Q: [6, 768], K: [6, 768]
# K.transpose(0, 1): [768, 6]
scores = torch.matmul(Q, K.transpose(0, 1))   # [6, 6]
```

**为什么这样写**：
- `scores[i][j]` 表示**第 $i$ 个词对第 $j$ 个词的原始相似度**
- 对角线 `scores[i][i]` 表示"自己对自己的关注度"

**第三步：缩放 + Softmax**

```python
d_k = Q.size(-1)   # 768
scaled_scores = scores / math.sqrt(d_k)      # 缩放
attn_weights = torch.softmax(scaled_scores, dim=-1)   # [6, 6]
```

**为什么这样写**：
- 除以 $\sqrt{d_k}$：当 $d_k$ 很大时，点积结果的数值会非常大，导致 Softmax 进入梯度饱和区（梯度趋近于 0，无法学习）。缩放将数值压缩到合理范围。
- `dim=-1`：对每一行做 Softmax，确保**每个词对所有词的权重之和为 1**。

**第四步：加权求和**

```python
# attn_weights: [6, 6], V: [6, 768]
output = torch.matmul(attn_weights, V)   # [6, 768]
```

**为什么这样写**：每个词的输出，是**所有词的 Value 向量的加权平均**。权重由 Query 和 Key 的匹配程度决定。输出形状与输入相同（$[6, 768]$），这是 Transformer 可以**堆叠多层**的基础。

---

## 第二章：Multi-Head Attention——多个视角同时观察

### 2.1 为什么需要多头

单头 Self-Attention 只有一个 $W^Q, W^K, W^V$，只能捕捉**一种类型**的语义关系。

**多头注意力（Multi-Head Attention）** 的核心思想：**将模型维度拆分成多个"头"，每个头独立计算注意力，让模型有机会同时捕捉多种不同类型的特征关联**。

> **说明**：上面"8 个专家分别关注语法、情感、实体关系"是**教学类比**，用于帮助建立直觉。实际上，训练后的注意力头到底学到了什么分工，**不可控也不可精确解释**，且强烈依赖数据、初始化和训练方式。不要把"头对应特定语义功能"当成事实。

### 2.2 核心公式

**整体多头注意力公式**：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

**单个注意力头计算公式**：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) = \text{softmax}\left(\frac{QW_i^Q (KW_i^K)^T}{\sqrt{d_k}}\right)VW_i^V$$

**符号说明**：
- $h$：注意力头的数量（如 8、12、16）
- $d_k$：单个头的特征维度，$d_k = d_{model} / h$（如 768/8 = 96）
- $W_i^Q, W_i^K, W_i^V$：第 $i$ 个头的投影矩阵
- $W^O$：输出投影矩阵，将拼接后的特征映射回原维度

### 2.3 维度变化全解析（含 Batch 维度）

假设参数：
- `batch_size = 2`
- `seq_len = 6`
- `d_model = 768`
- `n_heads = 8`
- `head_dim = 768 // 8 = 96`

| 步骤 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | $X$ | `[2, 6, 768]` | 2个样本，6个词，768维 |
| 线性投影 | $Q, K, V$ | `[2, 6, 768]` | 三个独立投影 |
| **分头** | $Q, K, V$ | `[2, 8, 6, 96]` | 8个头并行，每头96维 |
| 点积 | $QK^T$ | `[2, 8, 6, 6]` | 每个头独立的注意力分数 |
| 缩放 + Softmax | $\text{attn\_weights}$ | `[2, 8, 6, 6]` | 归一化权重 |
| 加权求和 | $\text{head\_output}$ | `[2, 8, 6, 96]` | 每个头的输出 |
| **拼接** | $\text{concat}$ | `[2, 6, 768]` | 8×96=768，拼回 |
| 最终投影 $W^O$ | $\text{output}$ | `[2, 6, 768]` | Multi-Head Attention 最终输出 |

**为什么这样写**：
- 分头后维度从 `[2, 6, 768]` → `[2, 8, 6, 96]`，让 PyTorch 可以通过广播机制**并行计算所有头**
- 拼接后通过 $W^O$ 投影，让模型学习如何融合多头的信息

---

## 第三章：PyTorch 手写 Multi-Head Attention

### 3.1 完整代码（工业级实现）

```python
import torch
import torch.nn as nn
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        dim:      模型维度 d_model（输入和输出的最后一维）
        heads:    注意力头数 h
        dim_head: 每个头的维度 d_k
        dropout:  正则化
        """
        super().__init__()
        
        # 在这个实现中，输出维度要回到 dim，所以要求：
        # 所有头拼接后的 inner_dim 必须与 dim 对齐。
        # 本代码用 dim_head 独立指定，因此显式要求 dim == heads × dim_head。
        assert dim == heads * dim_head, f"dim({dim}) 必须等于 heads({heads}) * dim_head({dim_head})"
        
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads   # 所有头拼接后的维度 = dim
        
        # 尺度缩放因子: 1 / sqrt(d_k)
        self.scale = dim_head ** -0.5
        
        # QKV 联合投影：常见高效实现，1 个线性层替代 3 个独立线性层
        # 输出维度 = inner_dim * 3（Q/K/V 拼接后），随后通过 chunk(3) 拆分
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        
        # Softmax 层
        self.attend = nn.Softmax(dim=-1)
        
        # 输出投影层 W^O
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        # Step 1: QKV 联合投影 + 拆分为 Q, K, V
        # to_qkv(x): [B, N, inner_dim * 3]
        # chunk(3, dim=-1): 拆成3个 [B, N, inner_dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # Step 2: 分头维度重排
        # 从 [B, N, h * d_k] 重排为 [B, h, N, d_k]
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )
        
        # Step 3: QK 点积 + 尺度缩放
        # q: [B, h, N, d_k], k.transpose: [B, h, d_k, N]
        # dots: [B, h, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Step 4: Softmax 归一化
        attn = self.attend(dots)   # [B, h, N, N]
        
        # Step 5: 注意力权重与 V 加权求和
        # attn: [B, h, N, N], v: [B, h, N, d_k]
        # out: [B, h, N, d_k]
        out = torch.matmul(attn, v)
        
        # Step 6: 拼接多头
        # 从 [B, h, N, d_k] 重排为 [B, N, h * d_k] = [B, N, inner_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Step 7: 最终投影 W^O
        return self.to_out(out)   # [B, N, dim]


# ========== 测试 ==========
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len = 16
    dim = 512
    heads = 8
    dim_head = 64   # 512 // 8 = 64
    
    # 创建模块
    mha = MultiHeadAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=0.1)
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = mha(x)
    
    print(f"输入维度:  {x.shape}")       # [2, 16, 512]
    print(f"输出维度:  {output.shape}")  # [2, 16, 512]
    assert x.shape == output.shape, "输入输出维度必须一致！"
    print("✅ 维度验证通过！")
```

### 3.2 代码逐段拆解

#### 初始化 `__init__`

```python
assert dim == heads * dim_head, f"dim({dim}) 必须等于 heads({heads}) * dim_head({dim_head})"
```

**为什么这样写**：本实现独立指定了 `dim_head`（每头维度），拼接后 `inner_dim = heads * dim_head`，必须与输入/输出维度 `dim` 严格相等，才能保证残差连接和后续层的形状一致。例如 `dim=512, heads=8` 时必须取 `dim_head=64`，不能出现不匹配的小数维度。（另一种经典写法是让 `dim_head = dim // heads`，此时断言为 `dim % heads == 0`。）

```python
self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
```

**为什么这样写**：这是**工业界常见的高效实现**。用 1 个线性层替代 3 个独立线性层，将输入直接映射为 Q/K/V 拼接后的维度（`inner_dim * 3`），后续通过 `chunk(3)` 拆分。融合后的矩阵乘法在主流硬件（GPU/TPU）上通常比 3 次独立乘法更快，属于常见的工程优化，不是硬性规定。

```python
self.scale = dim_head ** -0.5
```

**为什么这样写**：等价于 $1 / \sqrt{d_k}$，但用幂运算更高效，避免浮点数除法。

#### 前向传播 `forward`

**Step 1: QKV 投影与拆分**

```python
qkv = self.to_qkv(x).chunk(3, dim=-1)
```

- `to_qkv(x)`：`[B, N, dim]` → `[B, N, inner_dim * 3]`
- `chunk(3, dim=-1)`：拆成 3 个 `[B, N, inner_dim]`

**Step 2: 分头（关键！）**

```python
q, k, v = map(
    lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
    qkv
)
```

**为什么这样写**：
- `einops.rearrange` 让维度变换极其直观
- `'b n (h d) -> b h n d'` 表示：把最后一维 `(h d)` 拆开，变成独立的 `h` 和 `d` 维度
- 结果：`[B, N, h*d_k]` → `[B, h, N, d_k]`
- 这样每个头的 Q/K/V 就独立了，可以通过 PyTorch 广播**并行计算**

**如果不使用 einops**，等价的 `view + transpose` 写法：

```python
# 等价实现（不用 einops）
q = qkv[0].view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
k = qkv[1].view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
v = qkv[2].view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
```

**Step 3-5: 注意力计算**

```python
dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, h, N, N]
attn = self.attend(dots)                                    # [B, h, N, N]
out = torch.matmul(attn, v)                                # [B, h, N, d_k]
```

**为什么这样写**：
- `dots[b, h, i, j]`：第 $b$ 个样本、第 $h$ 个头中，第 $i$ 个位置对第 $j$ 个位置的相似度
- `attn`：Softmax 后的注意力权重，每行之和为 1
- `out`：每个头的独立输出

**Step 6-7: 拼接与投影**

```python
out = rearrange(out, 'b h n d -> b n (h d)')   # [B, N, inner_dim]
return self.to_out(out)                          # [B, N, dim]
```

**为什么这样写**：
- 拼接：把 `h` 个头的输出在特征维度上拼接
- 投影：`nn.Linear(inner_dim, dim)` 就是 $W^O$，将拼接后的特征映射回原输入维度

---

## 第四章：Masked Self-Attention——Decoder 的秘密武器

### 4.1 为什么需要 Mask

在 Transformer 的 **Decoder** 中，生成第 $i$ 个词时，**不能看到第 $i$ 个词之后的词**（否则就是作弊了）。

**Mask 的作用**：把"未来的信息"屏蔽掉，确保模型只能看到当前位置及之前的信息。

### 4.2 实现方式

```python
def forward(self, x, mask=None):
    # ... 前面的 QKV 计算相同 ...
    
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, h, N, N]
    
    # 如果有 mask，应用它
    if mask is not None:
        # mask: [B, 1, N, N] 或 [1, 1, N, N]
        # 被 mask 的位置设为 -inf，Softmax 后变为 0
        dots = dots.masked_fill(mask == 0, float('-inf'))
    
    attn = self.attend(dots)
    # ... 后续相同 ...
```

**为什么这样写**：
- `masked_fill(mask == 0, float('-inf'))`：把需要屏蔽的位置设为负无穷
- Softmax 中 $e^{-\infty} = 0$，所以被 mask 的位置权重为 0
- 这样当前位置就无法"关注"到未来的位置

**Mask 矩阵示例**（序列长度 4）：

```
      位0  位1  位2  位3
位0  [ 1    0    0    0 ]   ← 位0只能看自己
位1  [ 1    1    0    0 ]   ← 位1能看位0和位1
位2  [ 1    1    1    0 ]   ← 位2能看位0、1、2
位3  [ 1    1    1    1 ]   ← 位3能看所有
```

这是一个**下三角矩阵**，可以用 `torch.tril` 快速生成：

```python
def create_causal_mask(seq_len):
    """生成因果掩码（Causal Mask）"""
    mask = torch.tril(torch.ones(seq_len, seq_len))   # 下三角矩阵
    return mask.unsqueeze(0).unsqueeze(0)   # [1, 1, N, N]

mask = create_causal_mask(4)
print(mask)
```

---

## 第五章：交叉注意力（Cross-Attention）

### 5.1 与 Self-Attention 的区别

| 类型 | Q 的来源 | K/V 的来源 | 应用场景 |
|------|----------|-----------|----------|
| **Self-Attention** | 同一输入 $X$ | 同一输入 $X$ | Encoder、Decoder 内部 |
| **Cross-Attention** | 解码器输出 | 编码器输出 | Decoder 中连接 Encoder |

### 5.2 代码修改要点

将 `to_qkv` 拆分为 `to_q` 和 `to_kv`：

```python
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # 分开定义 Q 和 KV 的投影
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x_q, x_kv):
        """
        x_q:  [B, N_q, dim]    ← 来自 Decoder
        x_kv: [B, N_kv, dim]   ← 来自 Encoder
        """
        q = self.to_q(x_q)
        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        k, v = kv[0], kv[1]
        
        # 分头、计算注意力、拼接、投影（与 Self-Attention 相同）
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            (q, k, v)
        )
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

**为什么这样写**：Q 来自解码器（当前要生成的位置），K/V 来自编码器（源语言的信息）。这就是机器翻译中"解码器查编码器"的过程。

---

## 第六章：维度验证与调试技巧

### 6.1 完整维度验证代码

```python
def verify_dimensions():
    batch_size, seq_len = 2, 16
    dim, heads, dim_head = 512, 8, 64
    
    mha = MultiHeadAttention(dim, heads, dim_head)
    x = torch.randn(batch_size, seq_len, dim)
    
    # 手动跟踪每一步维度
    print(f"1. 输入 x:              {x.shape}")
    
    qkv = mha.to_qkv(x)
    print(f"2. QKV 拼接:            {qkv.shape}")
    
    q, k, v = qkv.chunk(3, dim=-1)
    print(f"3. Q/K/V 拆分:          {q.shape}, {k.shape}, {v.shape}")
    
    q = rearrange(q, 'b n (h d) -> b h n d', h=heads)
    print(f"4. Q 分头后:            {q.shape}")
    
    # K 也需要分头（与 Q 对称），再做 Q·Kᵀ
    k = rearrange(k, 'b n (h d) -> b h n d', h=heads)
    dots = torch.matmul(q, k.transpose(-1, -2))
    print(f"5. Q·Kᵀ 分数:           {dots.shape}")
    
    scaled = dots * mha.scale
    print(f"6. 缩放后:              {scaled.shape}")
    
    attn = mha.attend(scaled)
    print(f"7. Softmax 权重:        {attn.shape}")
    
    v = rearrange(v, 'b n (h d) -> b h n d', h=heads)
    out = torch.matmul(attn, v)
    print(f"8. 加权求和:            {out.shape}")
    
    out = rearrange(out, 'b h n d -> b n (h d)')
    print(f"9. 拼接后:              {out.shape}")
    
    final = mha.to_out(out)
    print(f"10. 最终输出:           {final.shape}")
    
    assert x.shape == final.shape
    print("\n✅ 所有维度验证通过！")

verify_dimensions()
```

### 6.2 常见维度错误排查

| 错误信息 | 原因 | 解决 |
|----------|------|------|
| `RuntimeError: matmul dim mismatch` | Q 的最后一维 ≠ K 的倒数第二维 | 检查 `transpose` 是否正确 |
| `assert dim % heads == 0 failed` | 模型维度不能被头数整除 | 调整 `dim` 或 `heads` |
| `view` 后形状不对 | `view` 的元素总数不匹配 | 用 `rearrange` 替代，更直观 |
| 输出维度 ≠ 输入维度 | 忘记 `to_out` 投影 | 确保最后有 `W^O` 投影 |

---

## 第七章：常见陷阱与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 忘记 Scale（不除以 √d_k） | **强烈推荐加 Scale**，否则 Softmax 输入过大易导致训练初期梯度不稳定或消失；这是 Transformer 论文标准设置 |
| dim 与 heads × dim_head 不匹配 | 像本实现这样独立指定 `dim_head` 时，确保 `dim == heads * dim_head`；经典实现也常取 `dim_head = dim // heads` |
| Softmax 维度搞错 | `dim=-1` 对每行做，确保每行和为 1 |
| 先写 3 个独立 `nn.Linear` 定义 QKV（性能稍差） | 工业界常见做法是用 1 个 `to_qkv` 融合投影（更高效），但两者在数学上等价，不是必须 |
| 注意力投影一定要开 / 一定要关 bias | bias 是否关闭属于**模型架构选择**：Llama 等部分现代 LLM 选择 `bias=False`，而 GPT-2 等模型的注意力投影带 bias；不属于硬性规定 |
| 忘记最后 `W^O` 投影 | 多头拼接后需要 `to_out`（即 W^O）映射回输入维度，否则残差连接维度不匹配 |
| 混淆 Self-Attention 和 Cross-Attention | Self: Q/K/V 同源；Cross: Q 来自解码器侧，K/V 来自编码器侧 |
| Decoder 自注意力不用 Causal Mask | **自回归生成场景下的 Decoder 自注意力需要因果 Mask**（下三角），防止模型看到未来 token；其他场景不一定需要 |
| 手动 `view + transpose` 导致维度混乱 | 用 `einops.rearrange`，维度变换更直观，不易出错 |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **Self-Attention** | $Q, K, V$ 同源，自己查自己 |
| **核心公式** | $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ |
| **Multi-Head** | $h$ 个头并行，每头 $d_k = d_{model}/h$ 维 |
| **整体公式** | $\text{MultiHead} = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$ |
| **QKV 联合投影** | 1 个 `nn.Linear(dim, inner_dim * 3)` 替代 3 个 |
| **分头** | `rearrange('b n (h d) -> b h n d')` |
| **拼接** | `rearrange('b h n d -> b n (h d)')` |
| **Causal Mask** | 下三角矩阵，防止 Decoder 看到未来 |
| **Cross-Attention** | Q 来自解码器，K/V 来自编码器 |

**养成的好习惯**：
1. 写注意力模块时，先在注释里列清楚每一步的**维度变化表**，确认前后对齐再写
2. **默认加上 Scale**（这是 Transformer 论文中的标准设置）；除非有明确理由，不要移除
3. 维度重排优先用 `einops.rearrange`，替代手动 `view + transpose`，避免维度混乱
4. 设计网络时让输入输出维度一致，确保可以轻松堆叠多层 Transformer Block
5. **自回归生成场景下的 Decoder 自注意力，加上 Causal Mask**

**下一步预告**：掌握了 Self-Attention 与 Multi-Head Attention 后，你将进入 **子词切分与分词器（Tokenizer）** 的学习——理解 BPE、WordPiece、SentencePiece 算法原理，掌握 HuggingFace Tokenizer 的加载与使用，熟悉 padding、truncation、max_length 等关键参数。这是连接"文本处理"与"Transformer 模型"的关键桥梁。继续加油！