# 学习包

## 块 1

```markdown
# KV Cache 机制

## 一、为什么需要 KV Cache？

### 1.1 自回归生成的重复计算

在大语言模型（LLM）的自回归生成过程中，每个新生成的token都依赖于之前所有已生成的token。假设已经生成了$t-1$个token，现在需要生成第$t$个token。如果不使用任何缓存机制，模型需要将完整的序列$[x_1, x_2, \ldots, x_{t-1}, x_t]$重新输入，计算每一层的Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中$Q, K, V$都是长度为$t$的序列。这意味着第$t$步的计算复杂度为$O(t^2)$（因为$QK^T$是$t \times t$的矩阵）。

随着序列长度$t$的增长，计算量呈平方级爆炸：

| 已生成长度 | 当前步复杂度 | 累计复杂度 |
|-----------|------------|-----------|
| 1 | $O(1)$ | $O(1)$ |
| 10 | $O(100)$ | $O(385)$ |
| 100 | $O(10^4)$ | $O(3.3 \times 10^5)$ |
| 1000 | $O(10^6

## 块 2

### 1.2 关键观察：历史 token 的 K 和 V 是不变的

在 Decoder 的 Causal Self-Attention 中，由于**因果掩码**（causal mask）的存在，每个 token 只能看到它自己及之前的 token，看不到未来的 token。（权威①）

这意味着：**已经生成的前 $t-1$ 个 token，它们的 Key 和 Value 向量不会受到后续生成的影响**。无论后面生成什么新 token，$k_1, k_2, \ldots, k_{t-1}$ 和 $v_1, v_2, \ldots, v_{t-1}$ 都保持不变。（①）

因此，我们可以将历史 token 的 K 和 V 缓存起来，避免重复计算。

## 二、KV Cache 的核心原理

（个性化提示：学习风格：hands-on）

## 块 3

### 2.1 KV 缓存机制

在 Transformer 模型中，KV 缓存用于优化注意力机制的计算效率。具体来说，KV 缓存存储的是**每一层 TransformerDecoder 中所有历史 token 的 Key 矩阵和 Value 矩阵**。通过缓存这些信息，模型可以避免重复计算，从而显著提升生成速度。

#### 为什么需要 KV 缓存？
1. **不缓存 Query**：每个新 token 的 Query 是全新的，无法从缓存中直接获取，因此需要重新计算。
2. **不缓存 Attention 输出**：Attention 输出

## 块 4

### 2.3 为什么只缓存 K 和 V？

这是初学者最常见的困惑。让我们从 Attention 的计算逻辑来理解：

- **Query** $q_t$：代表"当前 token 在问什么"，每个新 token 的问题都不同，必须重新计算
- **Key** $k_i$：代表"第 $i$ 个 token 回答什么主题"，一旦生成就不再改变
- **Value** $v_i$：代表"第 $i$ 个 token 的实际内容"，一旦生成就不再改变

Attention 的本质是"用当前的问题（Q）去匹配历史的主题（K），然后提取历史的内容（V）"。历史和主题不会变，所以 K 和 V 可以缓存；但每个新 token 的问题不同，所以 Q 必须重新计算。（①）

## 三、内存开销分析

KV Cache 的引入以**空间换时间**。理解它的内存占用，是分析 LLM 推理显存瓶颈的基础。（①）

（个性化提示：学习风格：hands-on）

## 块 5

# KV缓存机制的内存开销分析

## 3.1 通用计算公式

在标准多头注意力（MHA）中，KV缓存的内存占用可以通过以下公式计算：

$$\text{Memory}_{\text{per layer}} = \text{batch\_size} \times \text{seq\_len} \times \text{num\_key\_value\_heads} \times \text{head\_dim} \times 2 \times \text{bytes\_per\_elem}$$

其中：
- $\text{batch\_size}$ 表示一批样本的数量
- $\text{seq\_len}$ 表示序列长度
- $\text{num\_key\_value\_heads}$ 表

## 块 6

### 3.2 实例：`meta-llama/Llama-2-7b`（标准多头注意力）

以 `meta-llama/Llama-2-7b` 为例（标准多头注意力，`num_key_value_heads = num_heads = 32`，FP16，即 2 bytes/参数）：（①）

- $L = 32$ 层
- `num_key_value_heads = 32`，`head_dim = 128`，即 $d_{\text{model}} = 4096$
- 每 token 每层 K+V：$32 \times 128 \times 2\ \text{bytes} \times 2 = 16\ \text{KB}$
- 每 token 全部层：$32 \times 16\ \text{KB} = 512\ \text{KB}$

当 `batch_size = 64`，`seq_len = 2048` 时：

$$\text{Memory} = 64 \times 2048 \times 512\ \text{KB} = 64\ \text{GB}$$

而 `meta-llama/Llama-2-7b` 的模型参数本身仅占约 14 GB（FP16）。这意味着**KV Cache 的内存占用可以达到模型参数量的数倍**，是长上下文、大 batch 推理时的主要显存瓶颈。（①）

> **直观理解**：在 NVIDIA A100 40GB 显卡上，如果 batch_size=1、seq_len=4096，KV Cache 约需 2 GB，这看起来不多；但当 batch_size=16、seq_len=8192 时，KV Cache 就需要约 64 GB，远超单卡容量。

## 四、极简代码实现

以下是一个接近 Hugging Face Transformers 风格的极简实现，支持 MHA、GQA 和 MQA，帮助你理解 KV Cache 在代码中如何工作。（①）

（个性化提示：学习风格：hands-on）

## 块 7

```markdown
# KeyValue 缓存机制在标准多头注意力模型中的实现

## 1. KeyValue 缓存机制的重要性

KeyValue 缓存机制在模型推理中扮演着关键角色，尤其是在多头注意力机制中。通过缓存历史输入的 K 和 V，可以显著提升模型的推理效率。以下是 KeyValue 缓存机制的核心实现步骤。

## 2. KeyValue 缓存机制的实现

### 2.1 KeyValue 缓存的扩展

在标准多头注意力模型中，KeyValue 缓存机制通过扩展 K 和 V 的维度，将有限的 KV 头扩展到与 Q 头数相同的数量。具体实现如下：

```python
def repeat_kv(hidden_states, n_rep):
    """
    将 K/V 从 [batch, num_kv_heads, seq_len, head_dim] 
    重复扩展到 [batch, num_heads, seq_len, head_dim]，
    用于 GQA/MQA 场景。
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 先扩展维度，再 reshape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

### 2.2 多头注意力的计算

通过扩展后的 K 和 V，多头注意力的计算过程如下：

```python
class DecoderLayerWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads, num_key_value_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads or n_heads
        self.head_dim = d_model // n_heads
        
        # GQA/MQA 要求 n_heads 能被 num_key_value_heads 整除
        assert n_heads % self.num_key_value_heads == 0, \
            f"n_heads ({n_heads}) 必须能被 num_key_value_heads ({self.num_key_value_heads}) 整除"
        self.n_rep = n_heads // self.num_key_value_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        # GQA/MQA 中 Wk/Wv 的输出维度按实际 KV 头数计算
        kv_dim = self.num_key_value_heads * self.head_dim
        self.Wk = nn.Linear(d_model, kv_dim)
        self.Wv = nn.Linear(d_model, kv_dim)
        self.Wo = nn.Linear(d_model, d_model)
        
    def forward(self, x, past_kv=None):
        """
        x: [batch_size, seq_len, d_model]
        past_kv: tuple of (K, V), 每个 [batch_size, num_kv_heads, past_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算当前输入的 Q, K, V
        q = self.Wq(x)  # [batch, seq, d_model]
        k = self.Wk(x)  # [batch, seq, kv_dim]
        v = self.Wv(x)  # [batch, seq, kv_dim]

## 块 8

### 4.1 关键代码解读

1. **`repeat_kv` 函数**：GQA/MQA 的核心。将 K/V 从 `num_kv_heads` 通过维度扩展+reshape 重复到 `num_heads`，使 Q/K/V 头数对齐以计算 Attention。计算完成后，缓存中仍保存未扩展的 K/V，保持显存节省效果。
2. **`past_kv` 参数**：接收上一层的 `(K, V)` 缓存，形状为 `[batch, num_kv_heads, past_len, head_dim]`。
3. **`torch.cat` 拼接**：将历史 K/V 与当前新 token 的 K/V 沿序列长度维度拼接。
4. **Causal Mask**：`q` 中第 `i` 个 token 只能看到 `k` 中前 `past_len + i + 1` 个位置。当 `past_len=0` 时退化为标准下三角 mask；当 `seq_len=1`（逐 token 解码）时，当前 token 可看到全部历史位置。
5. **Prefill vs Decoding**：Prefill 阶段输入完整 prompt，Decoding 阶段每步只输入 1 个新 token。

## 五、工业级挑战：KV Cache 内存管理

KV Cache 虽然降低了计算复杂度，但引入了巨大的**内存管理挑战**。在生产环境中，这甚至比计算本身更棘手。（权威②）

（个性化提示：学习风格：hands-on）

## 块 9

### 5.1 现有系统的内存浪费

传统推理系统（如 FasterTransformer）为每个请求**预分配一段连续的 GPU 内存**，大小为该请求的最大可能序列长度（如 2048 tokens）。（权威②）

这导致三种内存浪费：

| 浪费类型 | 原因 | 后果 |
|---------|------|------|
| **Reserved（预留）** | 为未来 token 预留的槽位，在请求结束前不能被其他请求使用 | 占用大量空闲内存 |
| **Internal Fragmentation（内部碎片）** | 请求实际长度远小于最大长度，预分配的空间大部分未使用 | 内存利用率低 |
| **External Fragmentation（外部碎片）** | 不同请求预分配大小不同，释放后留下不连续的小块 | 无法分配给新请求 |

根据 vLLM 论文（权威②）在 NVIDIA A10G GPU、OPT-13B 模型、ShareGPT 数据集上的测试，现有系统中**实际有效内存利用率低至 20.4% - 38.2%**，超过 60% 的 KV Cache 内存被浪费。（权威②）

（个性化提示：学习风格：hands-on）

## 块 10

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

（个性化提示：学习风格：hands-on）

## 块 11

### 5.3 Copy-on-Write 与内存共享

在并行采样（一个 prompt 生成多个候选答案）时，所有候选共享相同的 prompt 部分的 KV Cache。vLLM 通过以下机制实现：（权威②）

1. Prompt 的 KV Block 被多个序列**共享**，引用计数 > 1
2. 当某个序列需要写入新 token 到共享块时，触发 **Copy-on-Write**：复制该块到新物理块，再写入
3. 原块的引用计数减 1

在束搜索（Beam Search）中，不同 beam 候选可以共享大部分前缀。根据 vLLM 论文（权威②）在相同测试条件下的实验，这可以节省 **37.6% - 55.2%** 的内存。（权威②）

### 5.4 调度与抢占

当 GPU 内存不足时，vLLM 支持两种策略将部分请求的 KV Cache 移出 GPU：（权威②）

- **Swapping（交换）**：将 KV Block 复制到 CPU 内存，需要时再搬回
- **Recomputation（重计算）**：直接丢弃 KV Cache，需要时重新计算（利用已生成的 token 作为新 prompt 快速重算）

## 六、其他优化方向

除了 PagedAttention，工业界还从多个角度优化 KV Cache：（①）

（个性化提示：学习风格：hands-on）

## 块 12

### 6.1 Multi-Query Attention (MQA)

所有 Query 头**共享同一组 K 和 V**。KV Cache 大小降为原来的 $1/\text{num\_heads}$。PaLM 系列（《PaLM: Scaling Language Modeling with Pathways》）、`tiiuae/falcon-7b` 等模型采用，在特定条件下可提升推理速度，但可能带来轻微质量损失。（①）

### 6.2 Grouped-Query Attention (GQA)

将 Query 头分组，每组共享一个 K、V 头。这是 MQA 和标准多头注意力的折中。不同模型的 GQA 配置差异较大，需以具体模型的 `config.json` 为准：（①）

- `meta-llama/Llama-2-7b`：`num_heads=32`，`num_key_value_heads=32`（标准多头注意力，**非 GQA**）
- `meta-llama/Llama-2-70b`：`num_heads=64`，`num_key_value_heads=8`（GQA，缓存大小降至 $1/8$）
- `meta-llama/Llama-3-8B`：`num_heads=32`，`num_key_value_heads=8`（GQA，缓存大小降至 $1/4$）

（个性化提示：学习风格：hands-on）

## 块 13

```markdown
# KV Cache 量化

## 什么是 KV Cache 量化？

在模型推理中，为了减少显存占用，我们可以通过 **量化** 技术对 KV Cache 进行优化。量化通过将高精度的权重和激活值转换为低精度表示（如 INT8、INT4 或 INT16），可以显著降低模型的显存占用。研究表明，在特定条件下，量化可以减少 50% 到 75% 的缓存占用，同时保持生成质量的可接受性（①）。

### 量化的作用
- **减少显存占用**：通过降低数据精度，模型在推理过程中所需的运行时内存（RAM）大幅减少。
- **提升运行效率**：低精度计算虽然不如浮点数计算精确，但在大多数应用场景下可以满足生成质量的要求（①）。

### 量化方法
- **INT8 量化**：将 32 位浮点数转换为 8 位整数，占用更少的内存。
- **INT4 量化**：进一步降低精度，占用更少的内存，但可能会影响

## 测评题目

### Q-KB0806-003
以下哪种技术不是用于推理加速的？

### Q-KB0806-004
FlashAttention 的核心思想是通过分块（Tiling）和重计算（Recomputation）来减少HBM（显存）的读写次数，从而加速并降低显存占用。（ ）

### Q-KB0805-003
以下哪种位置编码是**可学习**的（Learnable）？

### Q-KB0806-001
在自回归生成过程中，KV Cache的主要作用是？

### Q-KB0805-001
为什么Transformer需要引入位置编码（Positional Encoding）？

### Q-KB08-ST0807-002
推理时将模型切换到评估模式，应调用 `model.________()`。

### Q-KB0805-004
RoPE（旋转位置编码）通过将Query和Key向量进行旋转操作，使得注意力分数仅依赖于Token之间的**相对距离**。（ ）

### Q-KB08-ST0807-003
前向实验中 `input_ids` 通常表示？
