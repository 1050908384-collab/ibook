# 掌握 Transformer 的 Encoder-Decoder 结构

权威文章：《Attention Is All You Need》\
　　网址：https://arxiv.org/abs/1706.03762

① 底稿：《深度拆解 transformer 第02章：Transformer 架构全景图——Encoder、Decoder 与三种 Attention 的数据流》\
　　网址：https://blog.csdn.net/2604_95887441/article/details/160457248

> **适合人群**：已完成"环境准备：加载 GPT-2 进行前向传播实验"章节，观察过 GPT-2 的因果注意力掩码效果，理解 Self-Attention 核心概念的初学者。本节时长约 0.5 小时，目标是掌握 Transformer 的 Encoder-Decoder 整体架构，能够区分三种 Attention 的数据流，理解为什么 Decoder 需要 Masked Self-Attention 和 Cross-Attention。（①）
>
> **前置说明**：上节我们用 GPT-2 做了一个"解剖实验"，观察到它的注意力矩阵是一个下三角矩阵（上三角接近 0）——这就是**因果掩码（Causal Mask）**。当时我们只看到了 Decoder 的冰山一角。本节将回到 Transformer 的原始论文，系统讲解完整的 Encoder-Decoder 结构，以及三种 Attention 的分工。所有代码请在已安装 `transformers` 的 `pytorch` conda 环境中运行。

## 一、Encoder-Decoder 架构的本质（①）

Transformer 不是凭空发明的，它继承自**序列到序列（Seq2Seq）**任务的标准架构：

- **Encoder（编码器）**：把输入序列 $(x_1, ..., x_n)$ 编码成一组连续表示 $\mathbf{z} = (z_1, ..., z_n)$。Encoder 可以看到**完整的输入序列**（双向）。
- **Decoder（解码器）**：根据编码 $\mathbf{z}$，**自回归地**生成输出序列 $(y_1, ..., y_m)$。Decoder 在生成第 $t$ 个 token 时，只能依赖已生成的 $y_1...y_{t-1}$ 和 Encoder 输出，不能"偷看"未来的 token。（①）

> **关键特性**：Encoder 是双向的，Decoder 是单向（因果）的。这个区别是理解三种 Attention 的出发点。

## 二、Transformer 整体架构概览（①）

原论文中，Transformer 由两部分组成，各堆叠 $N=6$ 层：

```
输入 Embedding + 位置编码
        ↓
┌─────────────────────────────┐
│      Encoder (N=6 层)        │
│  ┌───────────────────────┐  │
│  │ Multi-Head Self-Attn  │  │  ← Q/K/V 都来自前一层
│  │     Add & Norm        │  │
│  ├───────────────────────┤  │
│  │    Feed Forward       │  │
│  │     Add & Norm        │  │
│  └───────────────────────┘  │
│           ... 重复6次        │
└─────────────────────────────┘
        ↓  Encoder 输出 z
        ↓─────────────────────────────┐
┌─────────────────────────────┐       │
│      Decoder (N=6 层)        │       │
│  ┌───────────────────────┐  │       │
│  │ Masked Multi-Head     │  │       │
│  │   Self-Attention      │  │  ← Q/K/V 来自前一层，带因果mask
│  │     Add & Norm        │  │       │
│  ├───────────────────────┤  │       │
│  │ Multi-Head Cross-Attn │  │  ← Q来自前一层，K/V来自Encoder输出z
│  │     Add & Norm        │  │◄──────┘
│  ├───────────────────────┤  │
│  │    Feed Forward       │  │
│  │     Add & Norm        │  │
│  └───────────────────────┘  │
│           ... 重复6次        │
└─────────────────────────────┘
        ↓
   Linear + Softmax
        ↓
   输出概率分布
```

> **图例说明**：每个子层后都有残差连接（Add）和 Layer Normalization（Norm）。Encoder 的输出 $\mathbf{z}$ 被送到**每一层** Decoder 的 Cross-Attention，而不是只送给第一层。（①）

## 三、Encoder 的层次结构（①）

Encoder 的每一层包含两个子层：

### 3.1 Multi-Head Self-Attention

- **Q/K/V 来源**：都来自 Encoder 前一层的输出
- **Mask**：无（双向，所有位置互相可见）
- **作用**：让每个位置聚合整个输入序列的上下文信息

### 3.2 Position-wise Feed-Forward Network (FFN)

两层全连接网络，中间用 ReLU 激活：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

维度变化：$d_{\text{model}}=512 \to d_{ff}=2048 \to d_{\text{model}}=512$

### 3.3 残差连接 + Layer Normalization

每个子层的输出都经过：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

**Encoder 的完整数据流**：（①）

$$
\begin{aligned}
x_1 &= \text{Embedding}(input) + \text{PositionalEncoding} \\
x_2 &= \text{LayerNorm}(x_1 + \text{MultiHeadAttention}(Q=x_1, K=x_1, V=x_1)) \\
x_3 &= \text{LayerNorm}(x_2 + \text{FFN}(x_2)) \\
&\text{...重复6次...} \\
\mathbf{z} &= x_{final} \quad \text{(Encoder输出)}
\end{aligned}
$$

## 四、Decoder 的层次结构（①）

Decoder 的每一层包含三个子层：

### 4.1 Masked Multi-Head Self-Attention

- **Q/K/V 来源**：都来自 Decoder 前一层的输出
- **Mask**：**因果掩码**（Causal Mask），防止看到未来位置
- **作用**：让 Decoder 聚合已生成序列的信息，保持自回归性质

**Mask 机制**：在计算 attention score 时，把未来位置的 score 设为 $-\infty$：

$$\text{score}_{ij} = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d_k}} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

经过 Softmax 后，未来位置的权重变为 0。（①）

**Mask 矩阵示例**（4 个位置）：

```
位置:    1    2    3    4
  1   [ ✓    ✗    ✗    ✗ ]
  2   [ ✓    ✓    ✗    ✗ ]
  3   [ ✓    ✓    ✓    ✗ ]
  4   [ ✓    ✓    ✓    ✓ ]
```

> 上节我们用 GPT-2 观察到注意力矩阵的上三角接近 0，正是这个因果掩码的效果。（①）

### 4.2 Multi-Head Cross-Attention

- **Q 来源**：Decoder 前一层
- **K/V 来源**：**Encoder 的输出** $\mathbf{z}$
- **Mask**：无（可以看到完整的输入序列）
- **作用**：让 Decoder 关注输入序列，是 Encoder 和 Decoder 之间的"桥梁"

> **重要**：Encoder 输出 $\mathbf{z}$ 被送到**每一层** Decoder 的 Cross-Attention（共 6 次），而不是只送给第一层。这确保每一层 Decoder 都能直接访问输入信息。（①）

### 4.3 Feed-Forward Network

与 Encoder 的 FFN 结构完全相同。

**Decoder 的完整数据流**：（①）

$$
\begin{aligned}
y_1 &= \text{Embedding}(output) + \text{PositionalEncoding} \\
y_2 &= \text{LayerNorm}(y_1 + \text{MultiHeadAttention}(Q=y_1, K=y_1, V=y_1, \text{mask}=\text{causal})) \\
y_3 &= \text{LayerNorm}(y_2 + \text{MultiHeadAttention}(Q=y_2, K=\mathbf{z}, V=\mathbf{z})) \\
y_4 &= \text{LayerNorm}(y_3 + \text{FFN}(y_3)) \\
&\text{...重复6次...} \\
\text{logits} &= \text{Linear}(y_{final}) \\
\text{probs} &= \text{Softmax}(\text{logits})
\end{aligned}
$$

## 五、三种 Attention 的系统对比（①）

三种 Attention 的公式都是 $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$，但 **Q/K/V 的来源不同**，导致作用完全不同。（①）

| Attention 类型 | 所在位置 | Q 来源 | K 来源 | V 来源 | Mask | 作用 |
|---|---|---|---|---|---|---|
| **Encoder Self-Attention** | Encoder 每层 | Encoder 前一层 | Encoder 前一层 | Encoder 前一层 | 无 | 编码输入序列的上下文 |
| **Decoder Masked Self-Attention** | Decoder 每层 | Decoder 前一层 | Decoder 前一层 | Decoder 前一层 | 因果 mask | 自回归生成，防止"看到未来" |
| **Decoder Cross-Attention** | Decoder 每层 | Decoder 前一层 | **Encoder 输出** | **Encoder 输出** | 无 | 让 Decoder 关注输入序列 |

> **面试常考点**：三种 Attention 都调用同一个 `MultiHeadAttention` 函数，区别只在于传入的 Q/K/V 参数不同。这体现了 Attention 机制的统一性——"用 Q 查询 K/V"。（①）

## 六、用 transformers 库验证架构（①）

我们可以通过 `transformers` 库加载一个 Encoder-Decoder 模型（如 T5），观察其内部结构：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载 T5-small（Encoder-Decoder 架构）
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

# 查看 Encoder 层数
print(f"Encoder 层数: {model.config.num_layers}")
# Encoder 层数: 6

# 查看 Decoder 层数
print(f"Decoder 层数: {model.config.num_decoder_layers}")
# Decoder 层数: 6

# 查看模型结构
print(model)
```

> **本地运行配置要求**：T5-small 模型约 240MB，首次加载需联网下载。CPU 即可运行，内存建议 ≥ 4GB。

### 6.1 观察 Encoder-Decoder 的前向传播

```python
import torch

# 输入文本
input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors="pt")

# 构造 Decoder 输入（翻译任务中，Decoder 的输入是目标语言的前缀）
decoder_input_ids = tokenizer("Das Haus", return_tensors="pt").input_ids

# 前向传播
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        output_attentions=True,
        output_hidden_states=True
    )

print(f"Encoder 最后一层隐藏状态形状: {outputs.encoder_last_hidden_state.shape}")
# torch.Size([1, 11, 512]) —— [batch, seq_len, hidden_dim]

print(f"Decoder 最后一层隐藏状态形状: {outputs.decoder_hidden_states[-1].shape}")
# torch.Size([1, 3, 512]) —— [batch, decoder_seq_len, hidden_dim]

print(f"Cross-Attention 数量: {len(outputs.cross_attentions)}")
# 6 —— 每层 Decoder 一个 Cross-Attention

print(f"每层 Cross-Attention 形状: {outputs.cross_attentions[0].shape}")
# torch.Size([1, 8, 3, 11]) —— [batch, n_heads, decoder_len, encoder_len]
```

> **关键观察**：`cross_attentions[0].shape` 的最后两个维度是 `[decoder_len, encoder_len]`，说明 Decoder 的每个位置都在关注 Encoder 的所有位置。这正是 Cross-Attention 的作用。（①）

> **本地运行配置要求**：此代码纯 CPU 运行即可，内存 ≥ 4GB。

## 七、架构演化：为什么后续模型"拆分"了 Encoder-Decoder？（①）

理解了完整的 Encoder-Decoder 结构后，一个自然的问题是：为什么 GPT-2、Llama、Qwen 这些大模型**只保留了 Decoder**？

### 7.1 Encoder-only：BERT（2018）

- **架构**：只用 Encoder，去掉 Decoder
- **训练方式**：Masked Language Model（MLM）
- **优势**：双向上下文，适合理解任务（分类、问答、NER）
- **劣势**：不适合生成任务

### 7.2 Decoder-only：GPT（2018-2024）

- **架构**：只用 Decoder，去掉 Encoder 和 Cross-Attention
- **训练方式**：Causal Language Model（Next Token Prediction）
- **优势**：
  - 架构更简单，参数量全部集中在 Decoder，更容易扩展到超大规模
  - 统一的训练目标，不需要成对的输入-输出数据
  - KV Cache 只需要缓存 Decoder 自己的 K 和 V，不需要额外缓存 Encoder 输出
- **劣势**：只能单向建模（但在生成任务上这不是问题）

### 7.3 Encoder-Decoder 并未消失（①）

虽然通用语言模型主流是 Decoder-only，但 Encoder-Decoder 在特定任务中仍然是 SOTA：

- **机器翻译**：T5、mBART
- **语音识别**：Whisper（音频 Encoder + 文本 Decoder）
- **多模态**：LLaVA（视觉 Encoder + 语言 Decoder）

> 后续"重点理解 Decoder-Only 架构"章节会深入讲解：为什么 GPT 去掉 Encoder 后仍然能工作？答案是——**把输入和输出拼接在一起**，用统一的 Causal Mask 处理。（①）

## 八、常见问题与解决（①）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 混淆三种 Attention 的 Q/K/V 来源 | 没有系统对比 | 记住核心区别：Self-Attention 的 Q/K/V 同源，Cross-Attention 的 K/V 来自 Encoder |
| 不理解为什么 Decoder 需要 Mask | 忽略了自回归性质 | 生成任务中，模型不能"偷看"未来的正确标签 |
| 以为 Encoder 输出只送到 Decoder 第一层 | 对数据流理解不深入 | Encoder 输出送到**每一层** Decoder 的 Cross-Attention |
| Cross-Attention 的 K 和 V 能否只用其一 | 对 Attention 机制理解不深入 | K 用于计算相似度，V 用于加权求和，缺一不可 |

## 九、总结与后续学习路线（①）

恭喜你完成了 Transformer Encoder-Decoder 结构的学习！你现在应该掌握了：

- ✅ Encoder-Decoder 架构的本质：Encoder 双向编码，Decoder 单向自回归生成
- ✅ Encoder 的层次结构：Self-Attention + FFN，残差连接 + LayerNorm
- ✅ Decoder 的层次结构：Masked Self-Attention + Cross-Attention + FFN
- ✅ 三种 Attention 的系统对比：Q/K/V 来源、Mask 有无、作用差异
- ✅ 因果 Mask 的数学机制：未来位置 score 设为 $-\infty$
- ✅ 用 T5 验证 Encoder-Decoder 的前向传播和数据流
- ✅ 架构演化视角：为什么 BERT 只用 Encoder，GPT 只用 Decoder

> **下一步预告**：在下一节 **"重点理解 Decoder-Only 架构（如 GPT/Llama/Qwen）"** 中，我们将深入讲解 Decoder-Only 模型的内部机制。你会理解：为什么去掉 Encoder 和 Cross-Attention 后，GPT 仍然能完成翻译、问答等任务？Causal Mask 如何在统一框架下处理"理解"和"生成"两种需求？以及 Decoder-Only 架构相比 Encoder-Decoder 在工程实现上的优势。现在，试着用上面的 T5 代码，观察 `cross_attentions` 矩阵，看看 Decoder 生成不同 token 时，分别关注了输入序列的哪些位置！