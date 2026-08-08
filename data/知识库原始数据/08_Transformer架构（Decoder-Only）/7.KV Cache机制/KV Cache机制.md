# KV Cache 机制

权威文章：《Attention Is All You Need》  
　　网址：https://arxiv.org/abs/1706.03762

权威文章：《Efficient Memory Management for Large Language Model Serving with PagedAttention》  
　　网址：https://arxiv.org/abs/2309.06180

① 底稿：《KV Cache 详解：大模型自回归生成的关键优化技术》  
　　网址：https://blog.csdn.net/zyw2002/article/details/161453041

② 参考文章：《推理优化：KV Cache 机制详解》  
　　网址：https://blog.csdn.net/csdddn/article/details/161789494

> **适合人群**：已完成"旋转位置编码（RoPE）"章节，理解 Decoder-Only 模型中 Attention、FFN、RMSNorm 各组件分工的初学者。本节时长约 1 小时，目标是理解 KV Cache 为什么能加速推理、掌握其内存开销的计算方法、能够手写极简实现，并了解工业界如何通过 PagedAttention 解决内存碎片问题。（①）
>
> **前置说明**：上节我们学习了 RoPE，知道了现代 LLM 如何在 Attention 中注入位置信息。但 Decoder-Only 模型在推理时还有一个核心问题——**自回归生成太慢**。生成第 100 个 token 时，为什么不需要重新计算前 99 个 token 的 Attention？KV Cache 正是解决这一问题的关键技术。本节将回答：KV Cache 缓存了什么？为什么只缓存 K 和 V 不缓存 Q？它的内存开销有多大？为什么生产环境需要 PagedAttention？所有代码请在已安装 `torch` 的 Python 环境中运行。

## 一、为什么需要 KV Cache？

### 1.1 自回归生成的重复计算

Decoder-Only 模型（如 GPT、Llama）在推理时采用**自回归生成**（auto-regressive generation）：逐个生成 token，每个新 token 都依赖之前所有已生成的 token。（权威①）

假设已经生成了 $t-1$ 个 token，现在需要生成第 $t$ 个 token。如果不使用任何缓存，模型需要将完整的序列 $[x_1, x_2, \ldots, x_{t-1}, x_t]$ 重新输入，计算每一层的 Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q, K, V$ 都是长度为 $t$ 的序列。这意味着第 $t$ 步的计算复杂度为 $O(t^2)$（因为 $QK^T$ 是 $t \times t$ 的矩阵）。（①）

随着序列增长，计算量呈平方级爆炸：

| 已生成长度 | 当前步复杂度 | 累计复杂度 |
|-----------|------------|-----------|
| 1 | $O(1)$ | $O(1)$ |
| 10 | $O(100)$ | $O(385)$ |
| 100 | $O(10^4)$ | $O(3.3 \times 10^5)$ |
| 1000 | $O(10^6)$ | $O(3.3 \times 10^8)$ |

不加缓存时，生成 1000 个 token 的累计计算量相当于重复计算了数百次完整的前向传播，推理速度极慢。（①）

### 1.2 关键观察：历史 token 的 K 和 V 是不变的

在 Decoder 的 Causal Self-Attention 中，由于**因果掩码**（causal mask）的存在，每个 token 只能看到它自己及之前的 token，看不到未来的 token。（权威①）

这意味着：**已经生成的前 $t-1$ 个 token，它们的 Key 和 Value 向量不会受到后续生成的影响**。无论后面生成什么新 token，$k_1, k_2, \ldots, k_{t-1}$ 和 $v_1, v_2, \ldots, v_{t-1}$ 都保持不变。（①）

因此，我们可以将历史 token 的 K 和 V 缓存起来，避免重复计算。

## 二、KV Cache 的核心原理

### 2.1 缓存什么？

KV Cache 缓存的是**每一层 Transformer Decoder 中，所有历史 token 的 Key 矩阵和 Value 矩阵**。（①）

- 不缓存 Query：每个新 token 的 Query 是全新的，必须重新计算
- 不缓存 Attention 输出：Attention 输出依赖于当前 Query 与所有历史 K 的交互，无法复用
- 只缓存 K 和 V：因为它们对历史 token 是静态的

### 2.2 工作流程

以单层 Self-Attention 为例，假设已经生成了 $t-1$ 个 token，缓存了：

$$K_{\text{cache}} = [k_1, k_2, \ldots, k_{t-1}]$$
$$V_{\text{cache}} = [v_1, v_2, \ldots, v_{t-1}]$$

**第 $t$ 步生成新 token $x_t$ 的过程**：（①）

1. 计算当前 token 的 $q_t, k_t, v_t$：
   $$q_t = x_t W_Q,\quad k_t = x_t W_K,\quad v_t = x_t W_V$$

2. 更新缓存：
   $$K_{\text{cache}} \leftarrow \text{concat}(K_{\text{cache}}, k_t)$$
   $$V_{\text{cache}} \leftarrow \text{concat}(V_{\text{cache}}, v_t)$$

3. 计算注意力：
   - 分数：$s_t = q_t \cdot [k_1, k_2, \ldots, k_t]^T$（长度为 $t$ 的行向量）
   - 权重：$a_t = \text{softmax}(s_t / \sqrt{d_k})$
   - 输出：$o_t = a_t \cdot V_{\text{cache}}$

4. 经过 FFN、RMSNorm 等后续层，得到输出 logits，采样得到下一个 token $x_{t+1}$

**复杂度变化**：（①）

- 不加缓存：第 $t$ 步复杂度为 $O(t^2)$，累计 $O(T^3)$
- 使用 KV Cache：第 $t$ 步复杂度为 $O(t)$（只需计算一个 $q_t$ 并与长度为 $t$ 的 K 做点积），累计 $O(T^2)$

> **重要提示**：第一个 token（Prefill 阶段）仍需计算完整序列的 K 和 V，这一步无法加速。KV Cache 的加速效果体现在后续的**逐 token 生成阶段**（Decoding 阶段）。

### 2.3 为什么只缓存 K 和 V？

这是初学者最常见的困惑。让我们从 Attention 的计算逻辑来理解：

- **Query** $q_t$：代表"当前 token 在问什么"，每个新 token 的问题都不同，必须重新计算
- **Key** $k_i$：代表"第 $i$ 个 token 回答什么主题"，一旦生成就不再改变
- **Value** $v_i$：代表"第 $i$ 个 token 的实际内容"，一旦生成就不再改变

Attention 的本质是"用当前的问题（Q）去匹配历史的主题（K），然后提取历史的内容（V）"。历史和主题不会变，所以 K 和 V 可以缓存；但每个新 token 的问题不同，所以 Q 必须重新计算。（①）

## 三、内存开销分析

KV Cache 的引入以**空间换时间**。理解它的内存占用，是分析 LLM 推理显存瓶颈的基础。（①）

### 3.1 计算公式

对于一层 Self-Attention，KV Cache 的内存占用为：

$$\text{Memory}_{\text{per layer}} = \text{batch\_size} \times \text{seq\_len} \times \text{num\_heads} \times \text{head\_dim} \times 2 \times \text{bytes\_per\_elem}$$

其中：
- $2$：同时缓存 K 和 V
- $\text{num\_heads} \times \text{head\_dim} = d_{\text{model}}$（模型隐藏层维度）

对于全部 $L$ 层：

$$\text{Memory}_{\text{total}} = \text{batch\_size} \times \text{seq\_len} \times L \times d_{\text{model}} \times 2 \times \text{bytes\_per\_elem}$$

### 3.2 实例：LLaMA 7B

以 LLaMA 7B 为例（FP16，即 2 bytes/参数）：（①）

- $L = 32$ 层
- $d_{\text{model}} = 4096$
- 每 token 每层 K+V：$2 \times 4096 \times 2\ \text{bytes} = 16\ \text{KB}$
- 每 token 全部层：$32 \times 16\ \text{KB} = 512\ \text{KB}$

当 `batch_size = 64`，`seq_len = 2048` 时：

$$\text{Memory} = 64 \times 2048 \times 512\ \text{KB} = 64\ \text{GB}$$

而 LLaMA 7B 的模型参数本身仅占约 14 GB（FP16）。这意味着**KV Cache 的内存占用可以达到模型参数量的 4 倍以上**，是长上下文、大 batch 推理时的主要显存瓶颈。（①）

> **直观理解**：在 A100 40GB 显卡上，如果 batch_size=1、seq_len=4096，KV Cache 约需 2 GB，这看起来不多；但当 batch_size=16、seq_len=8192 时，KV Cache 就需要约 64 GB，远超单卡容量。

## 四、极简代码实现

以下是一个接近 Hugging Face Transformers 风格的极简实现，帮助你理解 KV Cache 在代码中如何工作。（①）

```python
import torch
import torch.nn as nn
import math

class DecoderLayerWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        
    def forward(self, x, past_kv=None):
        """
        x: [batch_size, seq_len, d_model]
        past_kv: tuple of (K, V), each [batch_size, n_heads, past_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算当前输入的 Q, K, V
        q = self.Wq(x)  # [batch, seq, d_model]
        k = self.Wk(x)
        v = self.Wv(x)
        
        #  reshape 为多头: [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 如果有历史缓存，拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # 沿序列长度维度拼接
            v = torch.cat([past_v, v], dim=2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal Mask：确保只能看到当前及之前的位置
        # 当 seq_len=1（逐token生成）时，mask只需保证不看到未来，这里用下三角
        seq_len_total = k.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len_total), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(scores.device), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [batch, n_heads, seq, head_dim]
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.Wo(out)
        
        return out, (k, v)  # 返回输出和更新后的 KV Cache


# ========== 推理演示 ==========
d_model, n_heads, num_layers = 512, 8, 6
layers = [DecoderLayerWithKVCache(d_model, n_heads) for _ in range(num_layers)]

# 模拟已生成的 token（Prefill 阶段）
prompt_len = 10
x = torch.randn(1, prompt_len, d_model)  # batch=1

# 初始化 KV Cache
past_kvs = [None] * num_layers

# Prefill：计算 prompt 的 KV Cache
for i, layer in enumerate(layers):
    x, past_kvs[i] = layer(x, past_kv=past_kvs[i])

print(f"Prefill 后，每层 KV Cache 形状: {past_kvs[0][0].shape}")
# 输出: [1, 8, 10, 64] 即 [batch, heads, seq_len, head_dim]

# Decoding：逐 token 生成
for step in range(5):
    # 每步只输入最新生成的 1 个 token
    x_new = torch.randn(1, 1, d_model)
    
    for i, layer in enumerate(layers):
        x_new, past_kvs[i] = layer(x_new, past_kv=past_kvs[i])
    
    print(f"Step {step+1}: KV Cache 长度变为 {past_kvs[0][0].size(2)}")

# 最终 KV Cache 长度 = prompt_len + 5 = 15
```

> **本地运行配置要求**：纯 CPU 即可运行，内存 ≥ 4GB。

### 4.1 关键代码解读

1. **`past_kv` 参数**：接收上一层的 `(K, V)` 缓存，形状为 `[batch, n_heads, past_len, head_dim]`
2. **`torch.cat` 拼接**：将历史 K/V 与当前新 token 的 K/V 沿序列长度维度拼接
3. **Causal Mask**：即使使用缓存，也必须保证当前 token 只能 attend 到之前的位置
4. **Prefill vs Decoding**：Prefill 阶段输入完整 prompt，Decoding 阶段每步只输入 1 个新 token

## 五、工业级挑战：KV Cache 内存管理

KV Cache 虽然降低了计算复杂度，但引入了巨大的**内存管理挑战**。在生产环境中，这甚至比计算本身更棘手。（权威②）

### 5.1 现有系统的内存浪费

传统推理系统（如 FasterTransformer）为每个请求**预分配一段连续的 GPU 内存**，大小为该请求的最大可能序列长度（如 2048 tokens）。（权威②）

这导致三种内存浪费：

| 浪费类型 | 原因 | 后果 |
|---------|------|------|
| **Reserved（预留）** | 为未来 token 预留的槽位，在请求结束前不能被其他请求使用 | 占用大量空闲内存 |
| **Internal Fragmentation（内部碎片）** | 请求实际长度远小于最大长度，预分配的空间大部分未使用 | 内存利用率低 |
| **External Fragmentation（外部碎片）** | 不同请求预分配大小不同，释放后留下不连续的小块 | 无法分配给新请求 |

根据 vLLM 论文的测试，现有系统中**实际有效内存利用率低至 20.4% - 38.2%**，超过 60% 的 KV Cache 内存被浪费。（权威②）

### 5.2 PagedAttention：类比操作系统的虚拟内存

vLLM 提出的 **PagedAttention** 核心思想是：**将 KV Cache 分成固定大小的块（block），非连续存储，类比 OS 的虚拟内存分页**。（权威②）

核心设计：

1. **KV Block**：每个块包含固定数量 token 的 K 和 V（默认 16 个 token）
2. **逻辑块 vs 物理块**：请求的 KV Cache 是逻辑上连续的块序列，但物理上可以映射到 GPU 上任意空闲的物理块
3. **Block Table**：维护每个请求的逻辑块到物理块的映射表
4. **按需分配**：新 token 生成时才分配新物理块，而不是一开始就预分配最大长度

这带来了三个好处：（权威②）

- **消除内部碎片**：只分配实际需要的块，最后一块可能有少量浪费（最多 15 个 token）
- **消除外部碎片**：所有块大小相同，释放后可以被任意请求复用
- **支持内存共享**：通过引用计数和 Copy-on-Write，实现并行采样、束搜索等场景下的 KV Cache 共享

### 5.3 Copy-on-Write 与内存共享

在并行采样（一个 prompt 生成多个候选答案）时，所有候选共享相同的 prompt 部分的 KV Cache。vLLM 通过以下机制实现：（权威②）

1. Prompt 的 KV Block 被多个序列**共享**，引用计数 > 1
2. 当某个序列需要写入新 token 到共享块时，触发 **Copy-on-Write**：复制该块到新物理块，再写入
3. 原块的引用计数减 1

在束搜索（Beam Search）中，不同 beam 候选可以共享大部分前缀，vLLM 的实验显示这可以节省 **37.6% - 55.2%** 的内存。（权威②）

### 5.4 调度与抢占

当 GPU 内存不足时，vLLM 支持两种策略将部分请求的 KV Cache 移出 GPU：（权威②）

- **Swapping（交换）**：将 KV Block 复制到 CPU 内存，需要时再搬回
- **Recomputation（重计算）**：直接丢弃 KV Cache，需要时重新计算（利用已生成的 token 作为新 prompt 快速重算）

## 六、其他优化方向

除了 PagedAttention，工业界还从多个角度优化 KV Cache：（①）

### 6.1 Multi-Query Attention (MQA)

所有 Query 头**共享同一组 K 和 V**。KV Cache 大小降为原来的 $1/\text{num\_heads}$。PaLM、Falcon 等模型采用，推理速度提升明显，但可能带来轻微质量损失。（①）

### 6.2 Grouped-Query Attention (GQA)

将 Query 头分组，每组共享一个 K、V 头。这是 MQA 和标准多头注意力的折中。LLaMA 2/3 采用 GQA（如 7B 模型使用 32 个 Q 头、8 个 KV 头），缓存大小降至 $1/4$。（①）

### 6.3 KV Cache 量化

对 KV Cache 使用 INT8 甚至 INT4 量化。研究表明适度量化可以在保持生成质量的同时减少 50% - 75% 的缓存占用。（①）

### 6.4 滑动窗口缓存

只保留最近 $w$ 个 token 的 K、V，丢弃更早的信息。适用于局部依赖较强的任务，但会丧失长距离建模能力。（①）

## 七、常见问题与解决（①+②）

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| "为什么只缓存 K 和 V，不缓存 Q？" | 混淆 Q/K/V 的角色 | Q 代表"当前问题"，每个新 token 都不同；K/V 代表历史内容，生成后静态不变（①） |
| "KV Cache 会增加模型参数量吗？" | 混淆参数与激活/缓存 | KV Cache 是推理时的运行时内存，不是模型参数，不增加模型文件大小（①） |
| "Prefill 阶段为什么还是很慢？" | 期望全程加速 | Prefill 需要计算完整 prompt 的 K/V，无法避免；KV Cache 只加速 Decoding 阶段（①） |
| "为什么长上下文容易 OOM？" | 忽略内存公式 | KV Cache 随 seq_len 线性增长，长序列 + 大 batch 时显存爆炸，需用 PagedAttention 或量化（权威②） |
| "KV Cache 和模型参数哪个占显存多？" | 不了解具体数字 | 短序列时参数占主导；长序列（>2048）或大 batch 时 KV Cache 可能超过参数（①） |
| "MQA 和 GQA 会影响模型效果吗？" | 担心压缩损失 | 论文显示影响很小，LLaMA 2/3、Gemma 等主流模型均采用 GQA，效果依然优秀（①） |

## 八、总结与后续学习路线（权威①+②）

恭喜你完成了 KV Cache 的学习！你现在应该掌握了：

- ✅ **为什么需要 KV Cache**：自回归生成存在大量重复计算，不加缓存时复杂度为 $O(T^3)$
- ✅ **核心原理**：只缓存历史 token 静态不变的 K 和 V，每步只需计算新 token 的 Q，复杂度降为 $O(T^2)$
- ✅ **内存开销**：能够用公式估算任意模型的 KV Cache 显存占用，理解它是长上下文推理的主要瓶颈
- ✅ **代码实现**：理解 `past_key_values` 的工作模式，掌握 Prefill 和 Decoding 两阶段的分工
- ✅ **工业级优化**：理解 PagedAttention 如何通过块级内存管理解决碎片问题，以及 MQA/GQA/量化等压缩技术

> **下一步预告**：在下一节 **"前向传播实验（加载 GPT-2 验证理解）"** 中，我们将使用 Hugging Face Transformers 库加载真实的 GPT-2 模型，实际观察 `past_key_values` 的返回结构，验证我们对 Decoder-Only 架构、RoPE、KV Cache 的理解。现在，试着修改上面的代码，将 `batch_size` 设为 4，观察 KV Cache 的 shape 变化，并计算此时生成 1000 个 token 的预估显存占用！