# 学习包

## 块 1

## 二、原始 FFN 的“升维-激活-降维”三步走

### 步骤 1：升维线性变换

在动手写代码之前，先明确一个关键直觉：**Attention 负责让 token 之间“交流”，FFN 负责让每个 token 自己“变聪明”**。FFN 对每个 token 独立操作，不涉及 token 之间的交互。

原始 Transformer 的 FFN 公式为：

$$\text{FFN}(x) = \max\left( 0,xW_{1} + b_{1} \right)W_{2} + b_{2}$$

这个公式可以拆成三步，本块先聚焦**第一步：升维线性变换**。

---

**升维做了什么？**

输入 $x$ 的维度是 $d$（模型隐藏层维度），第一步计算：

$$xW_{1} + b_{1}$$

其中 $W_{1}$ 的形状是 $d \times d_{ff}$，$b_{1}$ 的形状是 $d_{ff}$。原始 Transformer 中 $d_{ff} = 4d$，即**先把维度从 $d$ 放大到 $4d$**。

**为什么要升维？** 升维为后续的激活函数提供了更丰富的特征空间。维度越高，每个 token 可被“重塑”的自由度越大，模型越有能力拟合复杂的语言模式。如果只做线性变换而不升维，FFN 的表达能力会被限制在线性空间，难以处理非线性关系。

---

**动手验证（PyTorch 最小示例）**

在已安装 `torch` 的 Python 环境中运行：

```python
import torch
import torch.nn as nn

d = 8          # 模型隐藏维度
d_ff = 4 * d   # 升维目标维度，原始 Transformer 取 4d

x = torch.randn(2, 3, d)   # 模拟 batch=2, seq_len=3, 每个 token 维度为 d
W1 = nn.Linear(d, d_ff)    # 升维线性层

out = W1(x)                # 形状变为 (2, 3, d_ff)
print(out.shape)           # torch.Size([2, 3, 32])
```

观察输出形状：每个 token 的维度从 8 变成了 32，这就是“升维”的直观体现。`nn.Linear` 内部已经包含了偏置 $b_1$，对应公式中的 $

## 块 2

### 步骤 1：升维线性变换

这一步的核心公式是：`z₁ = xW₁ + b₁`。

**它在做什么？** 输入 `x` 的形状是 `L × d_model`，其中 `L` 是序列长度，`d_model` 是每个 token 的特征维度。通过乘以升维矩阵 `W₁`（形状 `d_model × d_ff`），每个 token 的特征被投影到一个更高维的空间，维度从 `d_model` 变为 `d_ff`。这里 `d_ff = 4 × d_model`，例如 512 维会变成 2048 维。

**为什么需要升维？** 高维空间拥有更多的“特征方向”，相当于给每个 token 提供了更丰富的“表达画布”。后续的非线性变换（激活）可以在这个更广阔的空间里“雕刻”

## 块 3

**动手验证 ReLU 的作用**

先看公式：`z2 = max(0, z1)`。ReLU 把负数清零、正数保留，这就是“稀疏激活”。

**为什么必须要有它？** 如果去掉 ReLU，两个线性层 `z1 = xW1 + b1` 和 `z2 = z1W2 + b2` 可以合并成 `x(W1W2) + b1W2 + b2`——本质上还是一个线性变换，堆再多层也白搭。ReLU 的“分段线性”让 FFN 能用多个线性段逼近复杂函数，表达能力才真正起来。

**动手试一下**：拿一个小张量 `x`，分别计算 `x @ (W

## 块 4

### 3.1 ReLU 的局限

ReLU 计算简单，但在深层网络中藏着一个“暗坑”：**神经元死亡**。当某个神经元的输出长期落在负区间，梯度为 0，参数更新无法把它拉回正区间，它就会“躺平”——永久失活。注意：单次输出为负不算死亡，只有长期无法恢复才算。此外，负半轴信息被截断，导致**梯度稀疏**，梯度流不够平滑，训练效率打折。

### 3.2 GLU：门控机制的引入

GLU（门控线性单元）2016 年提出，核心是**门控机制**——像“智能水龙头”一样动态控制信息流量：

$$\text{GLU}(x) = \sigma\left( xW + b \right) \otimes \left( xV + c \

## 块 5

### 3.3 Swish：更平滑的激活

Swish 的公式是 `Swish(x) = x · σ(βx)`。当 β=1 时，PyTorch 里直接用 `F.silu()` 就能实现。和 ReLU 比，Swish 有三个关键优势：

- **负半轴非零梯度**：负数输入也有梯度，神经元不会彻底“死掉”
- **最大梯度约 1.1**：Swish 的导数最大约为 1.1，略高于 ReLU 的 1.0，但差异不大，主要优势在于平滑性和非零梯度
- **平滑非单调**：曲线不是单调上升，能更灵活地缩放输入

### 3.4 SwiGLU：门控 + Swish

SwiGLU 把 GLU 里的 Sigmoid 换成 Swish，形成了 Transformer FFN 的新范式：

`FFN_SwiGL

## 块 6

### 4.1 原始 FFN 的参数量

原始 FFN 只有两个权重矩阵：$W_{1}$（形状 $d \times 4d$）和 $W_{2}$（形状 $4d \times d$）。每个矩阵的参数量都是 $4d^{2}$，所以总参数量是：

$$4d^{2} + 4d^{2} = 8d^{2}$$

### 4.2 SwiGLU FFN 的参数量

SwiGLU 用三组权重矩阵替代了原来的两个：

- $W$（门控路径，形状 $d \times d_{\text{ff}}$）
- $V$（值路径，形状 $d \times d_{\text{ff}}$）
- $W_{2}$（输出投影，形状 $d_{\text{ff}} \times d$）

总参数量为：

## 块 7

## 四、SwiGLU 的维度设计：为什么是 \(\frac{8}{3} \cdot d\)？

### 4.3 令参数量相等

SwiGLU 内部有三个权重矩阵，参数量比原始 FFN 更多。为了保持模型容量和计算量不变，我们令两者参数量相等：

\[
3d \cdot d_{\text{ff}} = 8d^{2}
\]

解这个方程，得到：

\[
d_{\text{ff}} = \frac{8}{3}d \approx 2.67d
\]

这就是 \(\frac{8}{3} \cdot d\) 的由来。LLaMA 论文中采用 SwiGLU 激活函数，中间维度设为 \(\frac{2}{3} \cdot 4d\)，而非 PaLM 的 \(4d\)，用更小的维度换来了更强的表达能力。

### 4.4 工业

## 块 8

## 5.1 标准 SwiGLU FFN

你已经知道 Transformer 里 FFN 的作用，现在看它在 LLaMA 中如何用 SwiGLU 实现。核心就一行公式：

**`(SiLU(xW1) ⊗ (xW3)) W2`**

其中 ⊗ 表示逐元素相乘。拆开看三步：

1. **`xW1`**：把输入 `x` 线性变换到隐藏维度，作为“门控”信号。
2. **`F.silu(...)`**：对门控信号应用 SiLU 激活（即 Swish），让负值不完全归零，保留一点信息。
3. **`⊗ (xW3)`**：与另一路线性变换 `xW3` 逐元素相乘，实现“门控筛选”——只有门控放行的部分才保留。
4. **`W2`**：最后线性变换回原维度 `dim`，完成 FFN 输出。

代码里三个线性层分工明确：

- `w1`：门控（gate），输出到 `hidden_dim`
- `w3`：上投影（up），同样输出到 `hidden_dim`
- `w2`：下

## 块 9

**动手验证：SwiGLU 与 ReLU FFN 的参数量对比**

你已经知道 Transformer 里 FFN 有两种常见设计，这里我们直接跑代码，亲眼看看它们的参数量差多少。

**代码逻辑拆解**（跟着敲一遍）：

1. 设置输入维度 `dim = 4096`（这是模型隐藏层大小）。
2. 构建**原始 ReLU FFN**：先线性变换到 `4 * dim`（即 16384），过 ReLU，再线性变换回 `dim`。
3. 构建 **SwiGLU FFN**：隐藏层维度设为 `11008`（这是 LLaMA-7B 实际用的值）。
4. 分别统计两个网络的参数量（`p.numel()` 统计每个张量的元素个数，求和）。
5. 打印结果和比值。

**运行后你会看到**：

- ReLU FFN 参数量：`2 * 4096 * 16384

## 块 10

好的，我们直接进入正题。既然你喜欢动手实践，而且目标是理解Transformer架构，那我们就用最直观的方式，把SwiGLU这个“黑盒”拆开，亲手算一遍。

你之前的知识缺口在Transformer架构，而SwiGLU正是Transformer中FFN层的关键激活函数。理解它的计算流程，能帮你更扎实地掌握整个架构的数据流动。

---

### 5.3 迷你手算实例：亲手验证 SwiGLU 计算流程

理论看再多，不如动手算一次。我们用一个极小维度的例子，把SwiGLU的每一步都看清楚。别怕，这里只有简单的矩阵乘法和乘法。

**我们的“实验”设定**：
想象我们正在处理一个极其简单的数据点，它只有2个特征。
- 输入向量 `x = [1.0, 2.0]` （这就是我们当前token的特征表示）
- 为了演示，我们设定中间维度 `d_ff = 4`（实际模型中会大得多，这里只是为了方便手算）。

SwiGLU需要三个权重矩阵，我们给它们赋予具体的数值：
- 门控权重 `W`，形状是 (2, 4)：`[[0.5, 1.0, -0.5, 0.0], [1.5, -1.0, 0.5, 1.0]]`
- 值权重 `V`，形状是 (2, 4)：`[[-0.5, 2.0, 1.0, -1.0], [0.5, -

## 块 11

## 5.3 迷你手算实例：验证 SwiGLU 计算流程

你已理解 SwiGLU 的结构，现在通过一个极简例子，亲手走一遍计算，巩固“门控”直觉。

**设定**：设输入维度 $d=2$，隐藏维度 $d_{\text{ff}}=4$（为手算方便，非 $\frac{8}{3}d$）。输入向量 $x=[1, 2]$。

**步骤 1：升维**  
两个权重矩阵 $W_1, W_3$ 将 $x$ 从 2 维映射到 4 维。假设（仅演示）：
- $W_1 x = [1, 0, -1, 2]$（主路径）
- $W_3 x = [0.5, 1, -0.5, 1]$（门控路径）

**步骤 2：门控激活**  
对门控路径应用 SiLU（即 $\beta=1$ 的 Swish）：$\text

## 块 12

## 一、为什么需要位置编码？

Transformer 的核心是 Self-Attention，而 Attention 的计算本质是**加权平均**：

$$\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

关键问题：**不加入位置编码的 Self-Attention 对序列排列具有置换等变性**。如果你把序列中第 1 个 token 和第 10 个 token 交换位置，对应位置的输出也会随之交换，不是整个输出完全不变。这意味着模型无法感知"谁应该在哪个位置"这一顺序信息。

因此，如果没有位置编码，以下两句话对模型来说，仅就 Attention 计算本身而言，无法区分它们的顺序差异：

- "从 北京 到 上海 的 火车票"
- "从 上海 到 北京 的 火车票"

所以，必须引入某种机制让模型感知"谁在哪个位置"。

### 1.1 绝对位置编码：原始 Transformer 的选择

原始 Transformer 使用正弦/余弦函数生成位置编码，直接加到词嵌入上：

$$\text{PE}_{(pos,2i)} = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)$$

**动手验证一下**（在已安装 `torch` 的 Python 环境中运行）：

```python
import torch
import math

def sinusoidal_pe(pos, d_model):
    pe = torch.zeros(pos, d_model)
    for p in range(pos):
        for i in range(0, d_model, 2):
            pe[p, i] = math.sin(p / 10000 ** (2 * i / d_model))
            pe[p, i+1] = math.cos(p / 10000 ** (2 * i / d_model))
    return pe

# 生成 10 个位置、8 维的位置编码
pe = sinusoidal_pe(10, 8)
print(pe[0])  # 位置 0 的编码
print(pe[5])  # 位置 5 的编码
```

观察输出：每个位置都有**唯一**的编码向量，这就是"绝对位置"的含义——每个位置有自己固定的"身份证号"。

**关键理解**：这种编码方式是**加**到词嵌入上的（`x + PE`），而不是乘进去的。它给每个位置一个绝对标记，但模型

## 块 13

绝对位置编码：一句话版本是——给每个词的位置打一个“固定标签”，公式里用正弦和余弦函数（sin/cos）算出来。它简单，但有两个坑：

1. **外推性差**：训练时只见过 512 长度的序列，推理时给 1024 就懵了。
2. **不显式建模相对位置**：模型得自己猜“第 5 个词和第 8 个词距离 3”，而不是直接告诉它。

**为什么这是个问题？** 语言本质是相对位置——你关心“这个词离那个词多远”，而不是“这是第 500 个词”。理想情况是 Attention 的点积只依赖词本身和相对距离 m-n，与绝对位置无关

## 块 14

## 2.1 从 2D 情况开始理解

想象词嵌入维度 $d=2$，Query 和 Key 都是二维向量。RoPE 的核心思想一句话：

> **将词向量按其位置索引旋转，旋转角度与位置成正比。**

具体操作：
- 位置 $m$ 的 Query 向量 $q_m$ 逆时针旋转 $m\theta$ 角度
- 位置 $n$ 的 Key 向量 $k_n$ 逆时针旋转 $n\theta$ 角度

用复数表示（便于理解）：

$$f_q(x_m, m) = (W_q x_m) \cdot e^{im\theta}$$

$$f_k(x_n, n) = (W_k x_n) \cdot e^{

## 块 15

**动手验证：旋转后点积里只剩“距离”**

我们直接看旋转后 Query 和 Key 的点积公式：

$$\langle f_q(x_m,m), f_k(x_n,n) \rangle = \text{Re}\left[ (W_q x_m)(W_k x_n)^* \cdot e^{i(m - n)\theta} \right]$$

**关键观察**：结果中只出现了 $m - n$（相对位置），而 $m$ 和 $n$ 单独消失了！

**这意味着**：两个 token 相距 5 个位置，无论它们在序列的哪个绝对位置，它们的 Attention 权重模式是相同的。这正是“用绝对位置编码的方式实现相对位置编码”的精髓。

**动手检查**：

## 块 16

### 旋转矩阵的显式形式

前面我们用复数描述了旋转，现在把它变成代码里实际用的矩阵。对一个二维向量 $(x, y)$，旋转 $\theta$ 角度的变换是：

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**动手验证**：取向量 $X = \rho(\cos\phi, \sin\phi)$，乘上 $R(\theta)$ 后得到 $XR(\theta) = \rho(\cos(\phi + \theta), \sin(\phi + \theta))$。看，角度从 $\phi$ 变成了 $\phi + \theta$，这就是逆时针旋转。

记住两个关键性质，后面推导全靠它们：
1. **正交性**：$R(\theta)^T = R(-\theta)$，转置等于反向旋转。
2. **可加性**：$R(\theta_1)R(\theta_2) = R(\theta_1 + \theta_2)$，连续旋转就是角度相加。

**快速检查**：用 $R(30^\circ)$ 旋转向量 $(1, 0)$

## 块 17

### 频率参数：给每个维度定一个“转速”

你已经知道RoPE的核心是“旋转”，但高维空间怎么转？答案是把大空间拆成许多个2D小平面，每个小平面独立旋转。这里的关键就是**频率参数**——它决定每个小平面的旋转速度。

#### 频率公式

对于第 $k$ 个子空间（$k = 0, 1, \ldots, d/2 - 1$），频率定义为：

$$\theta_k = 10000^{-2k/d}$$

位置 $m$ 的旋转角度就是：

$$\text{angle}_{m,k} = m \cdot \theta_k$$

**直观理解**：$k$ 越大，$\theta_k$ 越小，旋转越慢。这就像时钟——时针转得慢（对应大 $k$），秒针转得快（对应小 $k$）。这个设计与原始Transformer的正弦编码一致，保证不同维度的旋转速度呈指数级变化。

#### 块对角旋转矩阵

有了频率，高维旋转矩阵 $R_{\Theta,m}^d$ 就是**块对角矩阵**——每个2D小块负责一个子空间的旋转：

$$R_{\Theta,m}^d = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0

## 块 18

## 3.3 高效计算：避免完整矩阵乘法

RoPE 的旋转矩阵是**块对角**的——只有相邻两维组成的小块参与旋转，其余位置都是零。这个稀疏结构意味着：**你根本不需要构造或乘一个大矩阵**，用逐元素操作就能完成同样的事。

看一个具体例子。对向量 \(x = (x_1, x_2, x_3, x_4, \ldots, x_{d-1}, x_d)\)，RoPE 的计算拆成三步：

1. **取余弦部分**：把 \(x\) 原样保留，逐元素乘以 \(\cos(m\theta_1), \cos(m\theta_1), \cos(m\theta_2), \cos(m\theta_2), \ldots\)——注意每个 \(\theta_i\) 对应的余弦值要**重复两次**，因为每个旋转块管两个维度。
2.

## 块 19

### 4.1 长距离衰减：距离越远，关联越弱

RoPE 通过设定旋转频率 $\theta_k = 10000^{-2k/d}$，实现了一个关键特性：**当两个 token 的相对距离 $|m - n|$ 增大时，它们内积的幅值会自然衰减**。这符合我们的直觉——距离越远的 token，语义关联应该越弱。RoPE 通过频率设计天然实现了这一点，无需额外约束。

### 4.2 与线性 Attention 兼容

标准 Attention 的复杂度是 $O(n^2)$，而线性 Attention 通过核技巧将其降至 $O(n)$。但大多数相对位置编码无法与线性 Attention 配合使用。RoPE 通过旋转注入位置信息，**保持了向量范数不变**，

## 块 20

## 5.1 极简版：理解核心逻辑

你已经掌握注意力机制，现在看 RoPE 如何把“位置”注入 Transformer。核心思想：**用旋转角度编码位置**，而不是直接加位置向量。

### 三步读懂代码

**第一步：准备旋转角度**

```python
theta = 10000 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
pos = torch.arange(seq_len, device=device).unsqueeze(1)
angles = pos * theta
```

- `theta` 是频率参数，维度 `head_dim//2`，每个维度对应不同旋转速度
- `pos` 是位置索引 `[0, 1, ..., seq_len-1]`
- `angles` 是每个位置、每个维度的旋转角度

**第二步：拆分奇偶索引**

```python
x1 = x[..., 0::2]  # 偶数索引
x2 = x[..., 1::2]  # 奇数索引
```

把 `head_dim` 分成两半，分别作为旋转的“横坐标”和“纵坐标”。

**第三步：应用 2D 旋转**

```python
rotated_x1 = x1 * cos - x2 * sin
rotated_x2 = x1 * sin + x2 * cos
```

这就是二维旋转公式：`(x1, x2)` 旋转 `angles` 角度后得到 `

## 块 21

**本块学习检查**：阅读下面的工业级 RoPE 实现，并对照注释，在代码旁标注出“缓存”和“广播”分别对应哪几行。

---

### 5.2 工业级版：支持缓存与广播

实际 LLM 推理中，RoPE 的 cos/sin 值会被反复使用，因此需要缓存。以下是接近 `meta-llama/Llama-2-7b` 实现风格的版本：

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

## 块 22

## 5.3 在 Attention 中使用 RoPE

你已经掌握了 RoPE 的基本原理，现在我们把旋转位置编码真正“装进”多头注意力里。下面这段代码就是完整的 `RoPEAttention` 模块，我们一步步拆开看。

### 整体结构

```python
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
```

**关键点**：
- `head_dim = dim // num_heads`：每个头的维度是总维度除以头数。
- `scale = head_dim ** -0.5`：缩放因子，防止点积过大。
- 四个线性层：Q/K/V 投影 + 输出投影。
- `self.rope` 是之前实现的旋转编码器，注意它接收的是 `head_dim`。

### forward 流程

```python
def forward(self, x):
    batch, seq_len, dim = x.shape
    
    # 1. 投影到 Q/K/V
    q = self.q_proj(x)  # [batch, seq_len, dim]
    k = self.k_proj(x)
    v = self.v_proj(x)
    
    # 2. 拆分为多头
    q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
```

**形状变化**：`[batch, seq_len, dim]` → `[batch, num_heads, seq_len, head_dim]`。`transpose(1, 2)` 把头的维度提到第 2 位，方便后续批量计算。

### 应用 RoPE

```python
    # 3. 应用 RoPE 到 Q 和 K（V 不需要位置编码）
    q = self.rope(q, seq_len)
    k = self.rope(k, seq_len)
```

**注意**：只有 Q 和 K 需要位置编码，V 不需要。因为位置信息是通过 Q 和 K 的点积注入的，V 只负责携带内容信息。

###

## 块 23

## 5.3 在 Attention 中使用 RoPE：为什么只旋转 Q 和 K？

在动手写代码之前，先抓住一个关键点：**RoPE 只应用于 Query 和 Key，不应用于 Value**。

为什么？因为 Attention 的权重是由 $QK^T$ 决定的——位置信息只需要影响“谁关注谁”的权重分配。Value 的角色是提供被提取的内容信息，它和位置无关。所以旋转 Q 和 K 就足够了，Value 保持原样。

## 5.4 验证：RoPE 的相对位置特性

现在动手验证 RoPE 的核心性质：**相同相对距离的点积应当相同，与绝对位置无关**。

把下面的代码复制到本地 Python 环境运行（纯 CPU 即可，内存 ≥ 4GB）：

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

## 块 24

## 5.5 长距离衰减可视化：用代码“看见”RoPE 的直觉

你已经在 Transformer 架构中理解了注意力机制——现在我们要用动手的方式，验证 RoPE 的一个关键性质：**距离越远，关联越弱**。

### 为什么值得动手跑一遍？

RoPE 不像其他位置编码那样需要额外学习参数，它通过旋转向量的方式把位置信息“编”进去。而“长距离衰减”正是它的优雅之处：**不需要任何额外约束，内积幅值会随相对距离自然下降**。这与你对注意力机制的理解直接挂钩——远距离 token 的关联本来就该更弱。

### 核心代码逐段拆解

```python
import torch

head_dim = 64
distances = [0, 1, 2, 4, 8, 16, 32, 64]

theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

这行代码生成频率系数 `theta`。`head_dim=64` 时，`torch.arange(0, 64, 2)` 产生 32 个偶数索引，每个对应一个频率。分母中的 `10000` 是 RoPE 的基频参数，它决定了不同维度旋转的快慢。

```python
x = torch.ones(1, head_dim // 2)
y = torch.ones(1, head_dim // 2)
```

这里把向量拆成两半（`head_dim // 2 = 32`），用两个全 1 向量表示一个单位向量。RoPE 的旋转操作是**成对**进行的——每两个维度构成一个旋转平面。

```python
for dist in distances:
    angles_0 = torch.zeros(1, head_dim // 2)
    angles_d = torch.tensor([[dist]]) * theta
    cos_0 = torch.cos(angles_0

## 块 25

## 七、总结与后续学习路线

恭喜你完成了 RoPE 的学习！针对你在 Transformer 架构上的学习目标，这里把核心要点串成一条清晰的链路，方便你快速内化。

### 你已掌握的核心内容

- **为什么需要位置编码**：Self-Attention 本身对序列顺序不敏感（排列等变性），必须显式注入位置信息，否则"我关注你"和"你关注我"的顺序就丢失了。
- **RoPE 的核心思想**：通过旋转矩阵把位置信息编码进 Q/K 向量，实现"绝对旋转，相对感知"——每个 token 的位置是绝对编码的，但 Attention 计算时只依赖相对距离 $m-n$。
- **数学推导主线**：2D 旋转 → 高维分块旋转 → 频率参数 $10000^{-2i/d}$ 设计 → 块对角矩阵的高效实现。频率参数与原始 Transformer 正弦编码一致，确保不同维度旋转速度呈指数级变化，覆盖多尺度位置信息。
- **关键性质**：长距离衰减、与线性 Attention 兼容、外推性。注意：理论上 RoPE 支持任意长度（角度连续），但实际外推受训练长度分布限制，超长外推需配合 NTK-aware、YaRN 等改进技术。
- **工业级实现**：缓存 cos/sin、广播机制、在 Multi-Head Attention 中的正确接入方式。

### 常见疑问速查（帮你扫清盲区）

| 疑问 | 一句话解答 |
|------|-----------|
| 为什么不加在 embedding 上？ | 加在 embedding 上退化为绝对编码，无法保证 $Q_m^T K_n$ 仅依赖 $m-n$ |
| 为什么 V 不旋转？ | Attention 权重由 $QK^T$ 决定，位置信息只需影响"谁

## 测评题目

### Q-KB08-ST0804-003
FFN 在 Transformer 中主要负责？

### Q-KB08-ST0804-002
SwiGLU 是一种带门控机制的________函数变体。

### Q-KB0805-002
原始Transformer论文中，使用的是基于________和________函数的固定位置编码。

### Q-KB0805-003
以下哪种位置编码是**可学习**的（Learnable）？

### Q-KB08-ST0807-002
推理时将模型切换到评估模式，应调用 `model.________()`。

### Q-KB0806-001
在自回归生成过程中，KV Cache的主要作用是？

### Q-KB0806-003
以下哪种技术不是用于推理加速的？

### Q-KB08-ST0803-002
RMSNorm 使用均方根统计进行归一化，通常不减去输入的________。
